"""Replay data/demo/readings.csv against a running API.

    python scripts/replay.py --url http://localhost:8000 [--shift-temperature 6]

With --shift-temperature the two temperature columns are raised by that many kelvin,
which should trigger a drift alert after one full window.
"""

import argparse
from pathlib import Path

import httpx
import pandas as pd

DEMO_CSV = Path(__file__).resolve().parent.parent / "data" / "demo" / "readings.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--shift-temperature", type=float, default=0.0)
    args = parser.parse_args()

    readings = pd.read_csv(DEMO_CSV).drop(columns=["machine_failure"])
    readings[["air_temperature_k", "process_temperature_k"]] += args.shift_temperature
    flagged = 0
    with httpx.Client(base_url=args.url, timeout=30) as client:
        for row in readings.to_dict("records"):
            resp = client.post("/predict", json=row)
            resp.raise_for_status()
            flagged += resp.json()["prediction"] == "failure"
        drift = client.get("/drift").json()
    print(f"sent {len(readings)} readings, {flagged} flagged as failure")
    print(f"drift detected: {drift['drift_detected']} {drift['drifted_features']}")


if __name__ == "__main__":
    main()
