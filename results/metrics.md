Held-out test set: 2000 rows, 68 failures (seed 42).
Precision, recall and F1 are for the failure class at threshold 0.5.

| model | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|
| always_normal | 0.000 | 0.000 | 0.000 | 0.034 |
| random_forest | 0.966 | 0.824 | 0.889 | 0.893 |
| xgboost_raw_features | 0.917 | 0.647 | 0.759 | 0.828 |
| xgboost_final | 0.931 | 0.794 | 0.857 | 0.890 |

Confusion matrix, final XGBoost model (rows are actual, columns are predicted):

| | predicted normal | predicted failure |
|---|---|---|
| actual normal | 1928 | 4 |
| actual failure | 14 | 54 |

5-fold cross-validation on the 8000 training rows:

| model | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|
| raw features | 0.833 | 0.609 | 0.704 | 0.756 |
| raw features + class weight | 0.517 | 0.804 | 0.629 | 0.728 |
| engineered features | 0.915 | 0.790 | 0.848 | 0.878 |
| engineered features + class weight | 0.781 | 0.841 | 0.810 | 0.870 |
