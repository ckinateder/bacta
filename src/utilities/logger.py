import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter for the logger."""

    # Color codes for different log levels (no underlining)
    grey = "\x1b[90m"
    blue = "\x1b[34m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Format string for different log levels
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    format_str = "%(name)s:%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: format_str + reset,
        logging.INFO: format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class CustomHandler(logging.StreamHandler):
    """Custom handler that uses the custom formatter."""

    def __init__(self, stream=None):
        super().__init__(stream)
        self.setFormatter(CustomFormatter())


# Create the top-level logger
logger = logging.getLogger("bacta")
logger.setLevel(logging.INFO)

# Add custom handler if no handlers are already configured
if not logger.handlers:
    handler = CustomHandler()
    logger.addHandler(handler)

# Prevent propagation to avoid duplicate logs
logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance. If no name is provided, returns the top-level logger.

    Args:
        name (str, optional): Name for the logger. Defaults to None.

    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        return logger
    else:
        return logging.getLogger(f"bacta.{name}")
