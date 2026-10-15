"""口令强度校验模块（P1-6）。

商用场景要求：
- 最小长度 8 位
- 至少包含大写、小写、数字各 1 个
- 可选特殊字符增强（推荐但不强制）
"""
import re
from typing import Optional


class PasswordStrengthError(Exception):
    """口令强度不足异常。"""
    pass


def validate_password_strength(password: str, min_length: int = 8) -> Optional[str]:
    """验证口令强度。
    
    Args:
        password: 待验证口令
        min_length: 最小长度（默认8）
    
    Returns:
        None 表示通过；返回字符串表示错误提示
    
    Raises:
        PasswordStrengthError: 口令强度不足
    """
    if not password:
        return "口令不能为空"
    
    if len(password) < min_length:
        return f"口令长度至少 {min_length} 位"
    
    # 检查是否包含大写字母
    if not re.search(r'[A-Z]', password):
        return "口令必须包含至少 1 个大写字母"
    
    # 检查是否包含小写字母
    if not re.search(r'[a-z]', password):
        return "口令必须包含至少 1 个小写字母"
    
    # 检查是否包含数字
    if not re.search(r'[0-9]', password):
        return "口令必须包含至少 1 个数字"
    
    # 通过所有检查
    return None


def validate_password_strength_strict(password: str) -> Optional[str]:
    """严格模式：额外要求特殊字符。
    
    用于高安全要求场景（管理员账户等）。
    """
    basic_error = validate_password_strength(password)
    if basic_error:
        return basic_error
    
    # 额外检查特殊字符
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', password):
        return "口令必须包含至少 1 个特殊字符"
    
    return None
