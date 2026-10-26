import httpx

from src import alerts, db
from src.config import get_settings
from tests.conftest import payload


def test_health_reports_model_and_database(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database"] is True


def test_predict_returns_probability_and_label(client, readings):
    resp = client.post("/predict", json=payload(readings.iloc[0].to_dict()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in ("normal", "failure")
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["model_name"] == get_settings().model_name


def test_prediction_is_logged_after_the_response(client, readings):
    row = payload(readings.iloc[1].to_dict()) | {"machine_id": "LOG-TEST"}
    request_id = client.post("/predict", json=row).json()["request_id"]
    with db.session_scope() as session:
        logged = session.query(db.PredictionLog).filter_by(request_id=request_id).one()
    assert logged.machine_id == "LOG-TEST"
    assert "power_w" in logged.features


def test_most_real_failures_in_the_sample_are_flagged(client, readings):
    failures = readings[readings["machine_failure"] == 1]
    flagged = sum(
        client.post("/predict", json=payload(row)).json()["prediction"] == "failure"
        for row in failures.to_dict("records")
    )
    assert flagged >= len(failures) // 2


def test_invalid_input_is_rejected(client, readings):
    bad = payload(readings.iloc[0].to_dict()) | {"product_type": "X"}
    assert client.post("/predict", json=bad).status_code == 422
    bad = payload(readings.iloc[0].to_dict()) | {"torque_nm": -5}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_predict(client, readings):
    rows = [payload(r) for r in readings.head(5).to_dict("records")]
    body = client.post("/predict/batch", json={"readings": rows}).json()
    assert len(body["results"]) == 5


def test_503_when_no_model_is_loaded(client, readings):
    predictor = client.app.state.predictor
    saved = predictor._model
    predictor._model = None
    try:
        resp = client.post("/predict", json=payload(readings.iloc[0].to_dict()))
    finally:
        predictor._model = saved
    assert resp.status_code == 503


def test_promote_and_rollback_switch_the_serving_version(client, trained):
    first, second = trained
    listing = client.get("/models").json()
    assert listing["champion_version"] == first.version

    promoted = client.post("/models/promote", json={"version": second.version}).json()
    assert promoted["champion_version"] == second.version
    assert client.get("/health").json()["model_version"] == second.version

    rolled = client.post("/models/rollback").json()
    assert rolled["champion_version"] == first.version
    assert client.get("/health").json()["model_version"] == first.version
    assert client.post("/models/rollback").status_code == 409


def test_promote_unknown_version_is_404(client):
    assert client.post("/models/promote", json={"version": "999"}).status_code == 404


def test_shifted_traffic_triggers_drift_alert_and_webhook(client, readings, monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(alerts.httpx, "post", fake_post)
    monkeypatch.setattr(get_settings(), "alert_webhook_url", "http://hook.test/x")

    window = get_settings().drift_window_size
    for row in readings.head(window).to_dict("records"):
        row = payload(row)
        row["air_temperature_k"] += 6
        row["process_temperature_k"] += 6
        client.post("/predict", json=row)

    status = client.get("/drift").json()
    assert status["drift_detected"]
    assert "air_temperature_k" in status["drifted_features"]
    assert len(calls) == 1
    assert calls[0]["alert_type"] == "data_drift"
