# src/utils/logger.py
import logging
import sys
import os
from pathlib import Path


def setup_logger(name: str, log_file: str | Path, level=logging.INFO):
    """
    配置全局 Logger
    :param name: Logger 名称
    :param log_file: 日志文件路径
    :param level: 日志级别
    """
    # 1. 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 获取 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 3. 清理已有的 Handlers (防止重复打印)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 格式化
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 5. FileHandler: 写入文件
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. StreamHandler: 输出到终端
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
