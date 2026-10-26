"""Write data/demo/readings.csv: 300 rows drawn from the held-out test split.

The demo replays these rows through the API. Using test rows keeps the demo
traffic separate from what the model was trained on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.training.pipeline import make_split  # noqa: E402

N_ROWS = 300


def main() -> None:
    settings = get_settings()
    split = make_split(settings.data_path, settings.test_size, settings.random_seed)
    sample = split.raw_test.sample(N_ROWS, random_state=settings.random_seed).copy()
    sample.insert(0, "machine_id", [f"M-{i:04d}" for i in range(len(sample))])
    cols = [
        "machine_id",
        "product_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "machine_failure",
    ]
    out = Path("data/demo/readings.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    sample[cols].to_csv(out, index=False)
    print(f"wrote {out}: {len(sample)} rows, {int(sample['machine_failure'].sum())} failures")


if __name__ == "__main__":
    main()
