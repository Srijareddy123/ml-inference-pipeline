# Results and details

Detail moved out of the README. Command: `make eval` (writes `results/metrics.md`).

## Setup

Held-out test split: 2000 of the 10000 rows, 68 of them failures. Failure class, threshold 0.5. A personal project for showing the full model lifecycle on public data.

## Model comparison

| model | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|
| always predict normal | 0.000 | 0.000 | 0.000 | 0.034 |
| random forest | 0.966 | 0.824 | 0.889 | 0.893 |
| XGBoost, raw features | 0.917 | 0.647 | 0.759 | 0.828 |
| **XGBoost, final** | 0.931 | 0.794 | 0.857 | 0.890 |

## Confusion matrix of the final model

Rows actual, columns predicted.

| | predicted normal | predicted failure |
|---|---|---|
| actual normal | 1928 | 4 |
| actual failure | 14 | 54 |

## Caveats

The random forest scores a little higher than XGBoost here. With 68 failures in the test set, a gap of a few cases is within noise, so I kept XGBoost as the served model. These numbers came from macOS arm64; a Linux run can differ in the last digit.

The AI4I data is synthetic, published by its author for research. It is not logs from a real plant.

## Cross-validation ablations

- **Engineered features.** Torque times speed (power), process minus air temperature, and tool wear times torque follow how the dataset defines its failure modes. Cross-validated F1 on the training rows went from 0.704 to 0.848 (`results/metrics.md`).
- **No class weighting.** Weighting the rare class raised recall to 0.841 but dropped precision to 0.781 and F1 to 0.810 in cross-validation, so I kept the unweighted model and the 0.5 threshold.

## How it works, in full

- Training splits 80/20 (stratified), builds features, fits XGBoost, and logs metrics, the confusion matrix and a sample of the training data to MLflow.
- A new version is promoted to the `champion` alias only if precision and recall pass a gate (0.70 and 0.60) and its F1 is not below the current champion. The old champion becomes `previous`.
- `POST /models/rollback` moves `champion` back to `previous`. `POST /models/promote` promotes any version. Both reload the model in the API.
- `/predict` returns the answer first. Writing the row to the database and the drift check run afterwards as FastAPI background tasks, so a database error never fails a prediction.
- Every 100 requests the KS test compares each feature in that window to the training sample of the serving model. A p-value below 0.05 divided by the number of features counts as drift. One alert per hour fires the webhook and writes a row.

## Design choices, in full

- **Aliases instead of MLflow stages.** Stages are deprecated. Two aliases (`champion`, `previous`) give promote and rollback with no extra state. Rollback clears `previous`, so a bad version is only re-promoted on purpose.
- **Bonferroni on the KS test.** Testing nine features at 0.05 each would raise false alarms on healthy traffic, so the threshold is divided by nine.
- **SQLite and MLflow files for the demo.** `make demo` needs no services. Docker Compose runs the same code on PostgreSQL and an MLflow server.

## Not done

This has never been deployed to SageMaker or any cloud. It is a plain FastAPI container. A SageMaker custom container would need `/ping` and `/invocations` on port 8080 and model loading from `/opt/ml/model`, none of which is implemented here.
