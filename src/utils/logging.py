"""Logging configuration helpers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure root logging for console and rotating file output.

    Args:
        log_dir: Directory where log files are written.
        level: Logging level name such as ``INFO`` or ``DEBUG``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Avoid duplicate handlers when invoked more than once.
        root_logger.setLevel(level)
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
