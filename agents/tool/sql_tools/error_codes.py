"""SQL error code classification for retry decisions.

Maps SQLState codes to (description, retryable) tuples.
Connection errors are retryable; syntax/auth/schema errors are not.
"""

# SQLState → (description, retryable)
SQL_ERROR_CODES: dict[str, tuple[str, bool]] = {
    # Connection errors — retryable
    "08001": ("连接建立失败", True),
    "08S01": ("连接中断", True),
    "08003": ("连接不存在", True),
    "08004": ("服务器拒绝连接", True),
    "08006": ("连接被关闭", True),
    "08001": ("连接异常", True),
    "08P01": ("连接协议违规", True),
    # Authentication errors — not retryable
    "28P01": ("密码错误", False),
    "28000": ("认证失败", False),
    "42501": ("权限不足", False),
    # Database errors — not retryable
    "3D000": ("数据库不存在", False),
    "42000": ("语法错误或数据库不存在", False),
    "3D070": ("Schema 不存在", False),
    # MySQL specific — not retryable
    "42S02": ("表不存在", False),
    "42S22": ("列不存在", False),
    "23000": ("约束违反", False),
    "HY000": ("通用错误", False),
}


def is_retryable(error_msg: str) -> bool:
    """判断 SQL 错误是否值得重试。

    连接类错误重试，语法/权限/表不存在类不重试。
    未知错误默认不重试（保守策略）。
    """
    if not error_msg:
        return False
    for code, (_desc, retryable) in SQL_ERROR_CODES.items():
        if code in error_msg:
            return retryable
    return False
