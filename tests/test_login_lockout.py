"""登录失败锁定：连续失败达到阈值后锁定，成功登录后重置。"""
from src.db.database import SessionLocal, init_db
from src.db.models import User
import src.team_auth as ta


def _make_user(db, username="u1", password="correct-pass"):
    user = User(
        username=username,
        email=f"{username}@x.com",
        is_active=True,
        role="viewer",
        password_hash=ta.hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_login_failure_locks_account(monkeypatch):
    init_db()
    monkeypatch.setattr(ta, "LOGIN_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(ta, "LOGIN_LOCKOUT_MINUTES", 15)
    db = SessionLocal()
    try:
        _make_user(db, "u1")
        assert ta.authenticate_credentials(db, "u1", "wrong-1") is None
        assert ta.authenticate_credentials(db, "u1", "wrong-2") is None
        assert not ta.account_locked(ta.find_active_user(db, "u1"))

        # 第三次失败触发锁定
        assert ta.authenticate_credentials(db, "u1", "wrong-3") is None
        assert ta.account_locked(ta.find_active_user(db, "u1"))

        # 锁定期间即使用对密码也被拒绝
        assert ta.authenticate_credentials(db, "u1", "correct-pass") is None
    finally:
        db.close()


def test_login_success_resets_counter(monkeypatch):
    init_db()
    monkeypatch.setattr(ta, "LOGIN_MAX_ATTEMPTS", 5)
    db = SessionLocal()
    try:
        _make_user(db, "u2")
        ta.authenticate_credentials(db, "u2", "wrong-1")
        ta.authenticate_credentials(db, "u2", "wrong-2")
        assert ta.find_active_user(db, "u2").failed_login_attempts == 2

        assert ta.authenticate_credentials(db, "u2", "correct-pass") is not None
        user = ta.find_active_user(db, "u2")
        assert user.failed_login_attempts == 0
        assert user.locked_until is None
    finally:
        db.close()


def test_reset_login_lockout_clears_state(monkeypatch):
    init_db()
    monkeypatch.setattr(ta, "LOGIN_MAX_ATTEMPTS", 2)
    db = SessionLocal()
    try:
        _make_user(db, "u3")
        ta.authenticate_credentials(db, "u3", "wrong-1")
        ta.authenticate_credentials(db, "u3", "wrong-2")
        user = ta.find_active_user(db, "u3")
        assert ta.account_locked(user)

        ta.reset_login_lockout(user)
        db.commit()
        user = ta.find_active_user(db, "u3")
        assert not ta.account_locked(user)
        assert user.failed_login_attempts == 0
    finally:
        db.close()
