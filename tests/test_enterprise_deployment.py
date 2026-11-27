from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_compose_enables_shared_rate_limit_and_cache():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["api"]["environment"]
    assert "REDIS_URL=redis://redis:6379/0" in env
    assert "RATE_LIMIT_STORAGE_URL=redis://redis:6379/0" in env
    assert "CACHE_STORAGE_URL=redis://redis:6379/0" in env
    assert "ZHIGE_LOCAL_OVERRIDES_PATH=/app/data/local_overrides.yaml" in env
    assert "ZHIGE_COOKIE_SECURE=${ZHIGE_COOKIE_SECURE:-1}" in env
    assert "ZHIGE_TOKEN_RESPONSE_ENABLED=${ZHIGE_TOKEN_RESPONSE_ENABLED:-0}" in env
    assert any("ZHIGE_BOOTSTRAP_TOKEN=" in item and ":?" in item for item in env)
    healthcheck = compose["services"]["api"]["healthcheck"]["test"]
    assert any("/api/health/live" in item for item in healthcheck)


def test_kubernetes_probes_and_cpu_defaults_are_consistent():
    deployment_text = (ROOT / "k8s" / "deployment.yaml").read_text(encoding="utf-8")
    deployment = next(
        doc
        for doc in yaml.safe_load_all(deployment_text)
        if doc and doc.get("kind") == "Deployment"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    assert container["livenessProbe"]["httpGet"]["path"] == "/api/health/live"
    assert container["startupProbe"]["httpGet"]["path"] == "/api/health/live"
    assert container["readinessProbe"]["httpGet"]["path"] == "/api/ready"
    assert "nvidia.com/gpu" not in deployment_text
    assert "gpu: \"true\"" not in deployment_text
    configmap = next(
        doc
        for doc in yaml.safe_load_all(
            (ROOT / "k8s" / "configmap.yaml").read_text(encoding="utf-8")
        )
        if doc and doc.get("kind") == "ConfigMap"
    )
    assert configmap["data"]["ZHIGE_LOCAL_OVERRIDES_PATH"] == "/data/local_overrides.yaml"


def test_ci_actually_runs_regression_tests():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python -m pytest tests" in ci


def test_runtime_images_include_postgres_backup_tools():
    for filename in ("Dockerfile", "Dockerfile.lite"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "postgresql-client" in text
