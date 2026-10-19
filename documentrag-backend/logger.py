"""统一日志模块：由 loguru 接管项目的全部日志输出。

- 控制台 + 按天轮转的文件输出（保留永久，不删除历史日志）
- 拦截标准库 logging（uvicorn / fastapi 等），统一交由 loguru 输出
- 其他模块直接 ``from logger import logger`` 使用即可
"""

import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from config import (
    LOG_BACKTRACE,
    LOG_CONSOLE_COLORIZE,
    LOG_CONSOLE_ENABLED,
    LOG_CONSOLE_LEVEL,
    LOG_DIAGNOSE,
    LOG_DIR,
    LOG_FILE_COMPRESSION,
    LOG_FILE_ENCODING,
    LOG_FILE_ENQUEUE,
    LOG_FILE_LEVEL,
    LOG_FILE_NAME,
    LOG_FILE_RETENTION,
    LOG_FILE_ROTATION,
    LOG_FORMAT,
    LOG_INTERCEPT_ENABLED,
    LOG_INTERCEPT_LOGGERS,
    LOG_INTERCEPT_ROOT_LEVEL,
)

__all__ = ["logger", "InterceptHandler", "setup_logger"]

_SELF_PATH = Path(__file__).resolve()
_LOGGING_DIR = Path(logging.__file__).resolve().parent


def _is_internal_frame(filename: str) -> bool:
    """判断调用帧是否属于 logging 内部或本模块，用于定位真正的日志调用点。"""
    try:
        path = Path(filename).resolve()
    except OSError:  # 极端情况下路径无法解析
        return False
    return path == _SELF_PATH or _LOGGING_DIR in path.parents or path.parent == _LOGGING_DIR


class InterceptHandler(logging.Handler):
    """将标准库 logging 的记录转发给 loguru，实现全量日志接管。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 跳过 logging 内部与本模块的调用帧，回溯到真正的日志调用位置
        frame, depth = logging.currentframe(), 0
        while frame is not None and _is_internal_frame(frame.f_code.co_filename):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _intercept_stdlib_logging(loggers: Iterable[str]) -> None:
    """用 loguru 接管根 logger 及指定第三方库 logger。"""
    handler = InterceptHandler()

    logging.basicConfig(
        handlers=[handler],
        level=LOG_INTERCEPT_ROOT_LEVEL,
        force=True,
    )

    for name in loggers:
        std_logger = logging.getLogger(name)
        std_logger.handlers = [handler]
        std_logger.propagate = False
        std_logger.setLevel(LOG_INTERCEPT_ROOT_LEVEL)


def setup_logger() -> None:
    """初始化 loguru：控制台输出 + 按天轮转的文件输出，并接管标准库日志。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.remove()

    if LOG_CONSOLE_ENABLED:
        logger.add(
            sys.stderr,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=LOG_CONSOLE_COLORIZE,
            backtrace=LOG_BACKTRACE,
            diagnose=LOG_DIAGNOSE,
        )

    logger.add(
        LOG_DIR / LOG_FILE_NAME,
        level=LOG_FILE_LEVEL,
        format=LOG_FORMAT,
        rotation=LOG_FILE_ROTATION,
        retention=LOG_FILE_RETENTION,
        compression=LOG_FILE_COMPRESSION,
        encoding=LOG_FILE_ENCODING,
        enqueue=LOG_FILE_ENQUEUE,
        backtrace=LOG_BACKTRACE,
        diagnose=LOG_DIAGNOSE,
    )

    if LOG_INTERCEPT_ENABLED:
        _intercept_stdlib_logging(LOG_INTERCEPT_LOGGERS)
