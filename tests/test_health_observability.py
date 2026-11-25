"""Health endpoint details must describe the selected runtime backend."""

from __future__ import annotations

from pathlib import Path

from api.routes.system import _disk_anchor_for, _llm_health_snapshot


def test_openai_health_uses_openai_model_and_never_returns_key():
    status, details = _llm_health_snapshot(
        {
            "backend": "openai",
            "model_name": "./models/local-qwen",
            "openai": {
                "model": "deepseek-chat",
                "base_url": "https://api.example.test/v1",
                "api_key": "secret-value",
            },
        }
    )

    assert status == "available"
    assert details == {
        "backend": "openai",
        "model": "deepseek-chat",
        "base_url": "https://api.example.test/v1",
        "api_key_configured": True,
    }
    assert "secret-value" not in repr(details)


def test_ollama_health_uses_ollama_model():
    status, details = _llm_health_snapshot(
        {
            "backend": "ollama",
            "ollama": {"model": "qwen3:4b", "base_url": "http://localhost:11434/v1"},
        }
    )

    assert status == "available"
    assert details["model"] == "qwen3:4b"
    assert details["base_url"] == "http://localhost:11434/v1"


def test_local_health_only_reports_cache_for_local_backend():
    status, details = _llm_health_snapshot(
        {"backend": "local", "model_name": "./models/qwen", "cache_dir": "models/hf"}
    )

    assert status == "available"
    assert details["model"] == "./models/qwen"
    assert "cache_dir" in details


def test_disk_anchor_uses_workspace_mount():
    target = Path.cwd() / "data" / "raw"
    assert _disk_anchor_for(target) == target.resolve().anchor
