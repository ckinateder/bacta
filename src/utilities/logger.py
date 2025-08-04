import logging
import datetime
from typing import Optional
import os


def create_logger(name: str = "bacta", level: int = logging.INFO) -> logging.Logger:
    # Get the logger (this will create it if it doesn't exist)
    logger = logging.getLogger(name)

    # Only add handler if the logger doesn't already have handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.setLevel(level)
        logger.addHandler(console_handler)

    return logger


# Global logger instance
_logger_instance: Optional[logging.Logger] = None


def get_log_level() -> int:
    """
    Get the log level from the environment variable LOG_LEVEL.
    """
    if os.getenv("LOG_LEVEL") == "DEBUG":
        return logging.DEBUG
    elif os.getenv("LOG_LEVEL") == "INFO":
        return logging.INFO
    elif os.getenv("LOG_LEVEL") == "WARNING":
        return logging.WARNING
    elif os.getenv("LOG_LEVEL") == "ERROR":
        return logging.ERROR
    elif os.getenv("LOG_LEVEL") == "CRITICAL":
        return logging.CRITICAL
    else:
        return logging.INFO


def get_logger(name: str = "bacta", level: int = get_log_level()) -> logging.Logger:
    """
    Get a logger instance. Creates a new one if it doesn't exist.

    Args:
        name: Name of the logger (default: "bacta")
        level: Logging level (default: environment variable LOG_LEVEL)

    Returns:
        Logger instance
    """
    # For the default logger name, use the global instance pattern
    if name == "bacta":
        global _logger_instance
        if _logger_instance is None:
            _logger_instance = create_logger("bacta", level)
        return _logger_instance
    else:
        # For other names, create/get the logger directly
        return create_logger(name, level)


def set_log_level(level: int = get_log_level()) -> None:
    """
    Set the global log level for the logger instance.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO,
               logging.WARNING, logging.ERROR, logging.CRITICAL). 
               Defaults to the environment variable LOG_LEVEL.

    Examples:
        >>> from src.utilities.logger import set_log_level
        >>> import logging
        >>> set_log_level(logging.DEBUG)  # Enable debug messages
        >>> set_log_level(logging.WARNING)  # Only show warnings and errors
    """
    global _logger_instance
    if _logger_instance is not None:
        _logger_instance.setLevel(level)
        # Also update the handler level to ensure all messages at the new level are processed
        for handler in _logger_instance.handlers:
            handler.setLevel(level)
    else:
        # If no logger instance exists yet, create one with the specified level
        _logger_instance = create_logger("bacta", level=level)
