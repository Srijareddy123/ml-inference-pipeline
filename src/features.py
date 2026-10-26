"""AI4I 2020 loading and feature building, shared by training and serving."""

import numpy as np
import pandas as pd

TARGET = "machine_failure"

RENAME = {
    "Type": "product_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": TARGET,
}

TYPE_CODES = {"L": 0, "M": 1, "H": 2}

BASE_FEATURES = [
    "product_type_code",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
ENGINEERED_FEATURES = ["temp_diff_k", "power_w", "wear_torque"]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

LABELS = {0: "normal", 1: "failure"}


def load_ai4i(path: str) -> pd.DataFrame:
    """Read the UCI csv and rename columns. The file starts with a byte order mark."""
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=RENAME)
    missing = {"product_type", TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def build_features(df: pd.DataFrame, engineered: bool = True) -> pd.DataFrame:
    """Turn raw sensor columns into the model matrix.

    The engineered columns follow the physics behind the dataset's failure modes:
    heat dissipation (process minus air temperature), mechanical power
    (torque times angular speed) and wear load (tool wear times torque).
    """
    out = pd.DataFrame(index=df.index)
    out["product_type_code"] = df["product_type"].map(TYPE_CODES).astype(float)
    for col in BASE_FEATURES[1:]:
        out[col] = df[col].astype(float)
    if engineered:
        out["temp_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
        out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * np.pi / 60
        out["wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    return out
