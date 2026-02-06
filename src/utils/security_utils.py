"""
安全工具模块
提供模型安全验证、输入清理等安全功能
"""

import hashlib
import pickle
import joblib
import os
from typing import Any, Optional


# def calculate_file_hash(file_path: str) -> str:
#     """计算文件的SHA256哈希值，用于完整性验证"""
#     sha256_hash = hashlib.sha256()
#     with open(file_path, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             sha256_hash.update(chunk)
#     return sha256_hash.hexdigest()


# def safe_load_pickle(file_path: str, expected_hash: Optional[str] = None) -> Any:
#     """
#     安全地加载pickle文件，可选择验证完整性

#     Args:
#         file_path: pickle文件路径
#         expected_hash: 期望的SHA256哈希值（可选）

#     Returns:
#         加载的对象

#     Raises:
#         ValueError: 如果哈希不匹配
#         FileNotFoundError: 如果文件不存在
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Pickle file not found: {file_path}")

#     # 如果提供了期望哈希，验证完整性
#     if expected_hash:
#         actual_hash = calculate_file_hash(file_path)
#         if actual_hash != expected_hash:
#             raise ValueError(
#                 f"文件完整性验证失败！期望哈希: {expected_hash}, 实际哈希: {actual_hash}"
#             )

#     try:
#         with open(file_path, "rb") as f:
#             return pickle.load(f)
#     except Exception as e:
#         raise ValueError(f"加载pickle文件失败: {e}")


# def safe_load_joblib(file_path: str, expected_hash: Optional[str] = None) -> Any:
#     """
#     安全地加载joblib文件，可选择验证完整性

#     Args:
#         file_path: joblib文件路径
#         expected_hash: 期望的SHA256哈希值（可选）

#     Returns:
#         加载的对象
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Joblib file not found: {file_path}")

#     # 如果提供了期望哈希，验证完整性
#     if expected_hash:
#         actual_hash = calculate_file_hash(file_path)
#         if actual_hash != expected_hash:
#             raise ValueError(
#                 f"文件完整性验证失败！期望哈希: {expected_hash}, 实际哈希: {actual_hash}"
#             )

#     try:
#         return joblib.load(file_path)
#     except Exception as e:
#         raise ValueError(f"加载joblib文件失败: {e}")


def sanitize_query(query: str) -> str:
    """
    清理查询字符串，防止潜在的注入攻击

    Args:
        query: 原始查询字符串

    Returns:
        清理后的查询字符串
    """
    if not query or not isinstance(query, str):
        return ""

    # 移除潜在的危险字符（根据实际需求调整）
    # 这里我们主要移除控制字符和特殊Unicode字符
    cleaned = "".join(char for char in query if ord(char) >= 32 or char in "\n\r\t")

    # 限制长度防止DoS攻击
    max_length = 10000  # 可根据实际需求调整
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    return cleaned


def validate_numeric_input(
    value: any, min_val: float = None, max_val: float = None
) -> float:
    """
    验证数值输入，防止除零和越界

    Args:
        value: 输入值
        min_val: 最小值限制
        max_val: 最大值限制

    Returns:
        验证后的数值

    Raises:
        ValueError: 如果验证失败
    """
    try:
        num = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"无效的数值输入: {value}")

    if min_val is not None and num < min_val:
        raise ValueError(f"数值 {num} 小于最小值 {min_val}")

    if max_val is not None and num > max_val:
        raise ValueError(f"数值 {num} 大于最大值 {max_val}")

    return num


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全的除法运算，防止除零错误

    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时返回的默认值

    Returns:
        除法结果
    """
    try:
        denominator = validate_numeric_input(denominator)
        if abs(denominator) < 1e-10:  # 接近零的检查
            return default
        return numerator / denominator
    except (ValueError, TypeError):
        return default
