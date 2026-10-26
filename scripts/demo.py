"""End-to-end demo on local files. No Docker, no network, no API keys. Run with `make demo`.

It trains three versions, shows the promotion gate, promotes and rolls back through
the API, replays sample traffic, then replays traffic with a simulated sensor
offset and shows the drift alert arriving at a local webhook.
"""

import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

STATE = ROOT / "demo_state"
shutil.rmtree(STATE, ignore_errors=True)
STATE.mkdir()

received: list[dict] = []


class Sink(BaseHTTPRequestHandler):
    """Stands in for a Slack or ops webhook."""

    def do_POST(self) -> None:  # noqa: N802 (name required by http.server)
        received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


sink = HTTPServer(("127.0.0.1", 0), Sink)
threading.Thread(target=sink.serve_forever, daemon=True).start()

# Settings are read at import time, so set them before importing the app.
os.environ.update(
    DATABASE_URL=f"sqlite:///{STATE / 'predictions.db'}",
    MLFLOW_TRACKING_URI=f"sqlite:///{STATE / 'mlflow.db'}",
    MLFLOW_ARTIFACT_ROOT=str(STATE / "artifacts"),
    ALERT_WEBHOOK_URL=f"http://127.0.0.1:{sink.server_port}/",
    LOG_LEVEL="WARNING",
)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import db  # noqa: E402
from src.api.main import app  # noqa: E402
from src.training.train import train  # noqa: E402

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def traffic(client: TestClient, frame: pd.DataFrame) -> int:
    failures = 0
    for row in frame.to_dict("records"):
        row.pop("machine_failure", None)
        resp = client.post("/predict", json=row)
        resp.raise_for_status()
        failures += resp.json()["prediction"] == "failure"
    return failures


def main() -> None:
    say("1. Train three versions on data/raw/ai4i2020.csv")
    variants = [
        ("v1 default", {}),
        ("v2 deeper trees", {"max_depth": 6}),
        ("v3 weak model", {"max_depth": 1, "n_estimators": 20}),
    ]
    for label, params in variants:
        r = train(params=params)
        m = r.metrics
        say(
            f"   {label}: precision {m['precision']:.3f} recall {m['recall']:.3f} "
            f"f1 {m['f1']:.3f} -> {r.reason}"
        )

    with TestClient(app) as client:
        champion = lambda: client.get("/models").json()["champion_version"]  # noqa: E731
        say(f"\n2. Registry: champion is v{champion()} after the gate")
        client.post("/models/promote", json={"version": "3"}).raise_for_status()
        say(f"   manual promote of v3 -> champion v{champion()}")
        client.post("/models/rollback").raise_for_status()
        say(f"   rollback -> champion v{champion()}")
        code = client.post("/models/rollback").status_code
        say(f"   second rollback -> HTTP {code} (nothing left to roll back to)")

        readings = pd.read_csv("data/demo/readings.csv")
        say(f"\n3. Replay {len(readings)} sample readings through POST /predict")
        flagged = traffic(client, readings)
        with db.session_scope() as session:
            logged = db.count_predictions(session)
        say(f"   flagged as failure: {flagged}, rows in prediction log: {logged}")
        say(
            f"   drift windows checked: {client.get('/drift').json()['last_window_checked']}, "
            f"alerts at webhook: {len(received)}"
        )

        drifted = readings.copy()
        drifted[["air_temperature_k", "process_temperature_k"]] += 6.0
        say(
            "\n4. Replay the same readings with +6 K on both temperatures (simulated sensor offset)"
        )
        traffic(client, drifted)
        status = client.get("/drift").json()
        say(
            f"   drift detected: {status['drift_detected']}, features: {status['drifted_features']}"
        )
        say(f"   alerts at webhook: {len(received)}")
        if received:
            say(f"   alert text: {received[0]['text']}")

    Path("results").mkdir(exist_ok=True)
    Path("results/demo_output.txt").write_text("\n".join(lines) + "\n")
    sink.shutdown()


if __name__ == "__main__":
    main()
