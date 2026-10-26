"""Settings read from environment variables or a git-ignored .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False, protected_namespaces=()
    )

    # Defaults point at local files so the demo needs no services. Docker Compose
    # overrides these with PostgreSQL and the MLflow server.
    database_url: str = "sqlite:///demo_state/predictions.db"
    mlflow_tracking_uri: str = "sqlite:///demo_state/mlflow.db"
    mlflow_artifact_root: str = "demo_state/artifacts"

    model_name: str = "machine_failure_xgboost"
    data_path: str = "data/raw/ai4i2020.csv"
    test_size: float = 0.2
    random_seed: int = 42

    # Promotion gate on the held-out split.
    min_precision: float = 0.70
    min_recall: float = 0.60

    # Drift detection
    drift_threshold: float = 0.05
    drift_window_size: int = 100
    enable_drift_detection: bool = True

    # Alerting
    alert_webhook_url: str = ""
    alert_cooldown_seconds: int = 3600
    enable_alerting: bool = True

    enable_prediction_logging: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
