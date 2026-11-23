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


def test_compose_requires_enterprise_bootstrap_password():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["api"]["environment"]
    assert any(item.startswith("AUTH_MODE=") for item in env)
    assert any("BOOTSTRAP_ADMIN_PASSWORD:?" in item for item in env)


def test_enterprise_stack_uses_central_storage_and_scales():
    docs = list(
        yaml.safe_load_all(
            (ROOT / "k8s/enterprise/stack.yaml").read_text(encoding="utf-8")
        )
    )
    by_kind = {}
    for item in docs:
        by_kind.setdefault(item["kind"], []).append(item)
    config = by_kind["ConfigMap"][0]["data"]
    deployment = by_kind["Deployment"][0]
    hpa = by_kind["HorizontalPodAutoscaler"][0]
    assert config["VECTOR_STORE_BACKEND"] == "pgvector"
    assert config["KG_BACKEND"] == "neo4j"
    assert config["AUTH_MODE"] == "enterprise"
    assert deployment["spec"]["replicas"] >= 3
    assert hpa["spec"]["minReplicas"] >= 3
    assert hpa["spec"]["maxReplicas"] > hpa["spec"]["minReplicas"]
