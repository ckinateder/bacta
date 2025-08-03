import logging
import datetime
from typing import Optional
import os


class CustomLogger:
    """
    Custom logger class that provides formatted logging output with timestamps,
    function names, and log levels.
    """

    def __init__(self, name: str = "iq_project", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler if it doesn't exist
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            # Create formatter
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)

            # Add handler to logger
            self.logger.addHandler(console_handler)

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)


# Global logger instance
_logger_instance: Optional[CustomLogger] = None


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


def get_logger(name: str = "iq_project", level: int = get_log_level()) -> CustomLogger:
    """
    Get a logger instance. Creates a new one if it doesn't exist.

    Args:
        name: Name of the logger
        level: Logging level (default: environment variable LOG_LEVEL)

    Returns:
        CustomLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CustomLogger(name, level)
    return _logger_instance


def set_log_level(level: int) -> None:
    """
    Set the global log level for the logger instance.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO,
               logging.WARNING, logging.ERROR, logging.CRITICAL)

    Examples:
        >>> from src.utilities.logger import set_log_level
        >>> import logging
        >>> set_log_level(logging.DEBUG)  # Enable debug messages
        >>> set_log_level(logging.WARNING)  # Only show warnings and errors
    """
    global _logger_instance
    if _logger_instance is not None:
        _logger_instance.logger.setLevel(level)
        # Also update the handler level to ensure all messages at the new level are processed
        for handler in _logger_instance.logger.handlers:
            handler.setLevel(level)
    else:
        # If no logger instance exists yet, create one with the specified level
        _logger_instance = CustomLogger(level=level)
