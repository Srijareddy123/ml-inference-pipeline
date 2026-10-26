"""Drift alerts: one row in the database and one webhook call, with a cooldown."""

import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy.orm import Session

from src.config import get_settings
from src.db import has_recent_alert, log_alert
from src.drift import DriftResult

logger = logging.getLogger(__name__)


def send_webhook(payload: dict) -> bool:
    """POST the alert as JSON. Never raises: alerting must not break serving."""
    url = get_settings().alert_webhook_url
    if not url:
        return False
    try:
        httpx.post(url, json=payload, timeout=5.0).raise_for_status()
        return True
    except httpx.HTTPError as exc:
        logger.warning("webhook failed: %s", exc)
        return False


def fire_drift_alert(
    session: Session, model_name: str, model_version: str, result: DriftResult
) -> bool:
    """Record the alert and call the webhook. Returns False when suppressed by the cooldown."""
    settings = get_settings()
    if not settings.enable_alerting:
        return False
    if has_recent_alert(session, model_name, settings.alert_cooldown_seconds):
        logger.info("drift alert suppressed by cooldown")
        return False
    payload = {
        "text": (
            f"Data drift on {model_name} v{model_version}: " f"{', '.join(result.drifted_features)}"
        ),
        "alert_type": "data_drift",
        "model_name": model_name,
        "model_version": model_version,
        "drifted_features": result.drifted_features,
        "p_values": {k: result.p_values[k] for k in result.drifted_features},
        "threshold": result.threshold,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    sent = send_webhook(payload)
    log_alert(
        session,
        {
            "model_name": model_name,
            "model_version": model_version,
            "drifted_features": result.drifted_features,
            "p_values": payload["p_values"],
            "threshold": result.threshold,
            "webhook_sent": sent,
        },
    )
    return True
