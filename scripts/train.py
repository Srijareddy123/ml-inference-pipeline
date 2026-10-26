"""Train and register a model. Run with `make train`; MLflow settings come from the environment."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--no-promote", action="store_true", help="register only, do not promote")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = {
        k: v
        for k, v in {"max_depth": args.max_depth, "n_estimators": args.n_estimators}.items()
        if v is not None
    }
    result = train(args.data_path, args.seed, params, promote=not args.no_promote)
    m = result.metrics
    print(
        f"version {result.version} run {result.run_id[:8]}: precision {m['precision']:.3f} "
        f"recall {m['recall']:.3f} f1 {m['f1']:.3f} -> {result.reason}"
    )


if __name__ == "__main__":
    main()
