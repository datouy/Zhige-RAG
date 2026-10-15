"""测试口令强度校验（P1-6）。"""
import pytest

from src.password_validator import validate_password_strength, validate_password_strength_strict


class TestPasswordValidation:
    """口令强度校验测试。"""

    def test_empty_password(self):
        """空口令应拒绝。"""
        assert validate_password_strength("") == "口令不能为空"
        assert validate_password_strength(None) == "口令不能为空"

    def test_too_short(self):
        """长度不足应拒绝。"""
        assert "长度至少" in validate_password_strength("Abc123")
        assert "长度至少" in validate_password_strength("aB1")

    def test_missing_uppercase(self):
        """缺少大写字母应拒绝。"""
        assert "大写字母" in validate_password_strength("abcdefg123")

    def test_missing_lowercase(self):
        """缺少小写字母应拒绝。"""
        assert "小写字母" in validate_password_strength("ABCDEFG123")

    def test_missing_digit(self):
        """缺少数字应拒绝。"""
        assert "数字" in validate_password_strength("Abcdefgh")

    def test_valid_basic_password(self):
        """符合基本要求的口令应通过。"""
        assert validate_password_strength("Abcd1234") is None
        assert validate_password_strength("Password123") is None
        assert validate_password_strength("MyPass2024") is None

    def test_valid_with_special_chars(self):
        """带特殊字符的口令应通过基本校验。"""
        assert validate_password_strength("Abc123!@#") is None
        assert validate_password_strength("Pass@word2024") is None

    def test_strict_mode_requires_special(self):
        """严格模式要求特殊字符。"""
        # 基本模式通过
        assert validate_password_strength("Abcd1234") is None
        # 严格模式拒绝
        assert "特殊字符" in validate_password_strength_strict("Abcd1234")
        # 带特殊字符应通过
        assert validate_password_strength_strict("Abcd123!") is None

    def test_custom_min_length(self):
        """自定义最小长度。"""
        assert validate_password_strength("Abc123", min_length=10) is not None
        assert validate_password_strength("Abc1234567", min_length=10) is None
