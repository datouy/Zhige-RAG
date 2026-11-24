"""密钥加密落盘：明文不进入 local_overrides.yaml，读取时解密。"""
import base64

import src.secret_crypto as sc
import src.settings_service as ss


def _set_key(monkeypatch):
    key = base64.urlsafe_b64encode(b"01234567890123456789012345678901")
    monkeypatch.setenv("ZHIGE_SECRETS_KEY", key.decode())
    sc._fernet = None
    sc._fernet_resolved = False


def test_encrypt_decrypt_roundtrip(monkeypatch):
    _set_key(monkeypatch)
    token = sc.encrypt_secret("sk-secret-value")
    assert token.startswith("enc:v1:")
    assert "sk-secret-value" not in token
    assert sc.decrypt_secret(token) == "sk-secret-value"


def test_placeholder_passthrough(monkeypatch):
    _set_key(monkeypatch)
    assert sc.encrypt_secret("${OPENAI_API_KEY}") == "${OPENAI_API_KEY}"
    assert sc.decrypt_secret("${OPENAI_API_KEY}") == "${OPENAI_API_KEY}"


def test_save_overrides_encrypts_and_load_decrypts(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    target = tmp_path / "local_overrides.yaml"
    monkeypatch.setattr(ss, "overrides_path", lambda: target)

    ss.save_overrides({
        "llm": {
            "backend": "openai",
            "openai": {"api_key": "sk-plain", "base_url": "https://x", "model": "gpt"},
        }
    })
    raw = target.read_text(encoding="utf-8")
    assert "sk-plain" not in raw
    assert "enc:v1:" in raw

    loaded = ss.load_overrides()
    assert loaded["llm"]["openai"]["api_key"] == "sk-plain"


def test_migrate_plaintext_secrets(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    target = tmp_path / "local_overrides.yaml"
    monkeypatch.setattr(ss, "overrides_path", lambda: target)
    target.write_text(
        "llm:\n  backend: openai\n  openai:\n    api_key: sk-old\n    base_url: https://x\n",
        encoding="utf-8",
    )

    assert ss.migrate_plaintext_secrets() is True
    raw = target.read_text(encoding="utf-8")
    assert "sk-old" not in raw
    assert "enc:v1:" in raw
