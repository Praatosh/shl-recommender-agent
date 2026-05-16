"""
SHL Assessment Recommender - Logging Setup
==========================================
Centralized logging configuration.
Uses structured logging with timestamps for production debugging.
"""

import logging
import sys
from app.config import get_settings


def setup_logging() -> logging.Logger:
    """
    Configure application-wide logging.
    Returns the root application logger.
    """
    settings = get_settings()

    # Create logger
    logger = logging.getLogger("shl_recommender")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on reload
    if logger.handlers:
        return logger

    # Console handler with structured format
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "shl_recommender") -> logging.Logger:
    """Get a named child logger."""
    return logging.getLogger(f"shl_recommender.{name}")
