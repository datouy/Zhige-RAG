"""企业凭据不得明文写入模型覆盖配置。"""
from __future__ import annotations

from cryptography.fernet import Fernet


def _configure_file_store(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRETS_BACKEND", "file")
    monkeypatch.setenv("ZHIGE_SECRETS_KEY", Fernet.generate_key().decode("ascii"))
    monkeypatch.setenv("SECRETS_FILE", str(tmp_path / "secrets.enc"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_encrypted_secret_roundtrip(monkeypatch, tmp_path):
    from src.secrets_store import get_secret, set_secret

    _configure_file_store(monkeypatch, tmp_path)
    secret = "sk-test-do-not-leak-123456"
    assert set_secret("llm.openai.api_key", secret) is True
    assert get_secret("llm.openai.api_key") == secret
    raw = (tmp_path / "secrets.enc").read_bytes()
    assert secret.encode() not in raw


def test_model_overrides_never_store_plaintext_key(monkeypatch, tmp_path):
    from src import settings_service

    _configure_file_store(monkeypatch, tmp_path)
    overrides = tmp_path / "local_overrides.yaml"
    monkeypatch.setattr(settings_service, "OVERRIDES_RELPATH", str(overrides))
    secret = "sk-test-model-key-987654"

    result = settings_service.update_model_settings(
        {
            "llm": {
                "backend": "openai",
                "openai": {
                    "base_url": "https://example.test/v1",
                    "model": "chat-model",
                    "api_key": secret,
                },
            }
        }
    )
    text = overrides.read_text(encoding="utf-8")
    assert secret not in text
    assert "api_key" not in text
    assert settings_service.get_secret("llm.openai.api_key") == secret
    assert result["llm"]["openai"]["api_key"] == "已设置"


def test_legacy_plaintext_override_is_scrubbed_and_migrated(monkeypatch, tmp_path):
    from src import settings_service

    _configure_file_store(monkeypatch, tmp_path)
    overrides = tmp_path / "local_overrides.yaml"
    overrides.write_text(
        "llm:\n"
        "  openai:\n"
        "    api_key: legacy-plaintext-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_service, "OVERRIDES_RELPATH", str(overrides))
    cleaned = settings_service.load_overrides()
    assert "api_key" not in cleaned["llm"]["openai"]
    assert "legacy-plaintext-secret" not in overrides.read_text(encoding="utf-8")
    assert settings_service.get_secret("llm.openai.api_key") == "legacy-plaintext-secret"
