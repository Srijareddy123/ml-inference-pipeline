"""Test setup: all state lives in a temp directory, nothing needs Docker or the network."""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="mlpipe-tests-"))
# Settings are read once at import, so this must run before any `src` import.
os.environ.update(
    DATABASE_URL=f"sqlite:///{_TMP / 'predictions.db'}",
    MLFLOW_TRACKING_URI=f"sqlite:///{_TMP / 'mlflow.db'}",
    MLFLOW_ARTIFACT_ROOT=str(_TMP / "artifacts"),
    ALERT_WEBHOOK_URL="",
    DRIFT_WINDOW_SIZE="50",
)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.training.train import train  # noqa: E402

DEMO_CSV = Path(__file__).resolve().parent.parent / "data" / "demo" / "readings.csv"


@pytest.fixture(scope="session")
def trained():
    """Version 1 is the champion, version 2 is registered but not promoted."""
    first = train()
    second = train(params={"max_depth": 6}, promote=False)
    return first, second


@pytest.fixture(scope="session")
def readings() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@pytest.fixture
def client(trained):
    with TestClient(app) as test_client:
        yield test_client


def payload(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "machine_failure"}
