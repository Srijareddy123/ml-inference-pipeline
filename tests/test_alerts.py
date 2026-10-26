import httpx
import pytest

from src import alerts, db
from src.config import get_settings
from src.drift import DriftResult


@pytest.fixture(autouse=True)
def tables():
    db.create_tables()


@pytest.fixture
def result():
    return DriftResult(True, ["torque_nm"], {"torque_nm": 0.001, "tool_wear_min": 0.4}, 0.005, 100)


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(alerts.httpx, "post", fake_post)
    monkeypatch.setattr(get_settings(), "alert_webhook_url", "http://hook.test/x")
    return calls


def test_alert_posts_to_webhook_and_logs_row(sent, result):
    with db.session_scope() as session:
        assert alerts.fire_drift_alert(session, "model-a", "3", result)
    assert len(sent) == 1
    body = sent[0][1]
    assert body["alert_type"] == "data_drift"
    assert body["drifted_features"] == ["torque_nm"]
    assert body["p_values"] == {"torque_nm": 0.001}
    with db.session_scope() as session:
        row = session.query(db.DriftAlert).filter_by(model_name="model-a").one()
        assert row.webhook_sent is True


def test_cooldown_suppresses_second_alert(sent, result):
    with db.session_scope() as session:
        assert alerts.fire_drift_alert(session, "model-b", "1", result)
    with db.session_scope() as session:
        assert not alerts.fire_drift_alert(session, "model-b", "1", result)
    assert len(sent) == 1


def test_webhook_failure_does_not_raise(monkeypatch, result):
    def boom(url, json, timeout):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(alerts.httpx, "post", boom)
    monkeypatch.setattr(get_settings(), "alert_webhook_url", "http://hook.test/x")
    with db.session_scope() as session:
        assert alerts.fire_drift_alert(session, "model-c", "1", result)
        row = session.query(db.DriftAlert).filter_by(model_name="model-c").one()
        assert row.webhook_sent is False


def test_empty_webhook_url_means_no_call(result):
    assert alerts.send_webhook({"text": "x"}) is False


def test_alerting_can_be_disabled(monkeypatch, result):
    monkeypatch.setattr(get_settings(), "enable_alerting", False)
    with db.session_scope() as session:
        assert not alerts.fire_drift_alert(session, "model-d", "1", result)
