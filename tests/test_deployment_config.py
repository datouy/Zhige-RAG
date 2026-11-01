from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def test_compose_uses_sync_postgres_driver():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["api"]["environment"]
    database_url = next(item for item in env if item.startswith("DATABASE_URL="))
    assert database_url.startswith("DATABASE_URL=postgresql+psycopg://")
    assert not any(item.startswith("SYNC_DATABASE_URL=") for item in env)


def test_k8s_single_writer_replicas():
    deployment = next(
        yaml.safe_load_all((ROOT / "k8s/deployment.yaml").read_text(encoding="utf-8"))
    )
    hpa = next(yaml.safe_load_all((ROOT / "k8s/hpa.yaml").read_text(encoding="utf-8")))
    assert deployment["spec"]["replicas"] == 1
    assert hpa["spec"]["maxReplicas"] == 1
