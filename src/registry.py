"""Model registry on MLflow with two aliases: `champion` (serving) and `previous`.

Promote moves `champion` to a version and remembers the old one as `previous`.
Rollback moves `champion` back to `previous` and clears it, so a bad version can
only be re-promoted on purpose.
"""

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

CHAMPION = "champion"
PREVIOUS = "previous"


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    run_id: str

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"


class Registry:
    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def _by_alias(self, name: str, alias: str) -> ModelInfo | None:
        try:
            mv = self.client.get_model_version_by_alias(name, alias)
        except MlflowException:
            return None
        return ModelInfo(name=name, version=str(mv.version), run_id=mv.run_id)

    def champion(self, name: str) -> ModelInfo | None:
        return self._by_alias(name, CHAMPION)

    def previous(self, name: str) -> ModelInfo | None:
        return self._by_alias(name, PREVIOUS)

    def list_versions(self, name: str) -> list[dict[str, Any]]:
        champion, previous = self.champion(name), self.previous(name)
        rows = []
        for mv in self.client.search_model_versions(f"name='{name}'"):
            version = str(mv.version)
            alias = (
                CHAMPION
                if champion and champion.version == version
                else PREVIOUS
                if previous and previous.version == version
                else None
            )
            rows.append({"version": version, "alias": alias, "run_id": mv.run_id})
        return sorted(rows, key=lambda r: int(r["version"]))

    def promote(self, name: str, version: str) -> ModelInfo:
        try:
            mv = self.client.get_model_version(name, version)
        except MlflowException as exc:
            raise RegistryError(f"{name} version {version} does not exist") from exc
        current = self.champion(name)
        if current and current.version == str(version):
            return current
        if current:
            self.client.set_registered_model_alias(name, PREVIOUS, current.version)
        self.client.set_registered_model_alias(name, CHAMPION, str(version))
        return ModelInfo(name=name, version=str(version), run_id=mv.run_id)

    def rollback(self, name: str) -> ModelInfo:
        previous = self.previous(name)
        if previous is None:
            raise RegistryError(f"{name} has no previous version to roll back to")
        self.client.set_registered_model_alias(name, CHAMPION, previous.version)
        self.client.delete_registered_model_alias(name, PREVIOUS)
        return previous

    def run_metric(self, run_id: str, metric: str) -> float | None:
        return self.client.get_run(run_id).data.metrics.get(metric)
