"""PostgreSQL (or SQLite in demo mode) storage for prediction logs and drift alerts."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    machine_id: Mapped[str] = mapped_column(String(64), index=True)
    features: Mapped[dict] = mapped_column(JSON)
    prediction: Mapped[str] = mapped_column(String(32))
    failure_probability: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(32))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DriftAlert(Base):
    __tablename__ = "drift_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    drifted_features: Mapped[list] = mapped_column(JSON)
    p_values: Mapped[dict] = mapped_column(JSON)
    threshold: Mapped[float] = mapped_column(Float)
    webhook_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True)


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables() -> None:
    Base.metadata.create_all(get_engine())


def db_ok() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def log_prediction(session: Session, row: dict[str, Any]) -> None:
    session.add(PredictionLog(**row))


def count_predictions(session: Session) -> int:
    return len(session.scalars(select(PredictionLog.id)).all())


def has_recent_alert(session: Session, model_name: str, cooldown_seconds: int) -> bool:
    cutoff = _utcnow() - timedelta(seconds=cooldown_seconds)
    stmt = select(DriftAlert.id).where(
        DriftAlert.model_name == model_name, DriftAlert.created_at > cutoff
    )
    return session.scalars(stmt.limit(1)).first() is not None


def log_alert(session: Session, row: dict[str, Any]) -> DriftAlert:
    alert = DriftAlert(**row)
    session.add(alert)
    session.flush()
    return alert
