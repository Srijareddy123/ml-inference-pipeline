import pytest
from mlflow.tracking import MlflowClient

from src.registry import Registry, RegistryError


@pytest.fixture
def registry(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    reg = Registry(uri)
    client = MlflowClient(tracking_uri=uri)
    client.create_registered_model("m")
    for _ in range(3):
        client.create_model_version("m", source=str(tmp_path), run_id="r")
    return reg


def test_promote_sets_champion_and_remembers_previous(registry):
    registry.promote("m", "1")
    registry.promote("m", "2")
    assert registry.champion("m").version == "2"
    assert registry.previous("m").version == "1"


def test_promote_same_version_twice_keeps_previous(registry):
    registry.promote("m", "1")
    registry.promote("m", "2")
    registry.promote("m", "2")
    assert registry.previous("m").version == "1"


def test_rollback_restores_previous_and_clears_it(registry):
    registry.promote("m", "1")
    registry.promote("m", "3")
    assert registry.rollback("m").version == "1"
    assert registry.champion("m").version == "1"
    assert registry.previous("m") is None


def test_rollback_without_previous_raises(registry):
    registry.promote("m", "1")
    with pytest.raises(RegistryError):
        registry.rollback("m")


def test_promote_unknown_version_raises(registry):
    with pytest.raises(RegistryError):
        registry.promote("m", "99")


def test_list_versions_shows_aliases(registry):
    registry.promote("m", "1")
    registry.promote("m", "2")
    aliases = {row["version"]: row["alias"] for row in registry.list_versions("m")}
    assert aliases == {"1": "previous", "2": "champion", "3": None}
