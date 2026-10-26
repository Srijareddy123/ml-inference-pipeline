import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURES, TARGET, build_features, load_ai4i

CSV = Path("data/raw/ai4i2020.csv")


def test_dataset_has_expected_shape_and_failure_count():
    df = load_ai4i(str(CSV))
    assert len(df) == 10000
    assert int(df[TARGET].sum()) == 339


def test_committed_csv_matches_download_script_checksum():
    from scripts.download_data import SHA256

    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == SHA256


def test_build_features_adds_physical_features():
    df = pd.DataFrame(
        [
            {
                "product_type": "H",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 100.0,
            }
        ]
    )
    out = build_features(df)
    assert list(out.columns) == FEATURES
    row = out.iloc[0]
    assert row["product_type_code"] == 2
    assert row["temp_diff_k"] == 10.0
    assert np.isclose(row["power_w"], 40.0 * 1500.0 * 2 * np.pi / 60)
    assert row["wear_torque"] == 4000.0


def test_build_features_without_engineering_keeps_base_columns_only():
    df = load_ai4i(str(CSV)).head(3)
    assert "power_w" not in build_features(df, engineered=False).columns
