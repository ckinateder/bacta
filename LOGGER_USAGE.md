# Custom Logger Usage Guide

This project includes a custom logging system that provides colored output and centralized logging configuration. The logger is instantiated in `src/__init__.py` and can be used across all files in the project.

## Features

- **Colored Output**: Different log levels have different colors for easy identification
- **Centralized Configuration**: All logging is configured in one place
- **Module-Specific Loggers**: Create loggers for specific modules or classes
- **Configurable Log Levels**: Easy to change log levels at runtime
- **Timestamp Formatting**: Consistent timestamp format across all logs

## Quick Start

### Basic Usage

```python
from src import logger, get_logger

# Use the top-level logger
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

# Create a module-specific logger
module_logger = get_logger("my_module")
module_logger.info("Module-specific message")
```

### In Your Files

Add this to the top of any file where you want to use logging:

```python
from src import get_logger

# Get a logger for this module
logger = get_logger("your_module_name")
```

## Log Levels

The logger supports standard Python logging levels:

- **DEBUG**: Detailed information for debugging
- **INFO**: General information messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical error messages

### Setting Log Level

```python
from src import set_log_level

# Set to DEBUG to see all messages
set_log_level("DEBUG")

# Set to WARNING to see only warnings and above
set_log_level("WARNING")
```

## Color Coding

- **DEBUG**: Grey
- **INFO**: Blue
- **WARNING**: Yellow
- **ERROR**: Red
- **CRITICAL**: Bold Red

## Examples

### Class with Logger

```python
from src import get_logger

class MyClass:
    def __init__(self, name):
        self.name = name
        self.logger = get_logger(f"my_class.{name}")
        self.logger.info(f"Initialized {name}")
    
    def do_work(self):
        self.logger.info("Starting work")
        # ... do work ...
        self.logger.info("Work completed")
```

### Exception Handling

```python
from src import logger

try:
    # Some risky operation
    result = 10 / 0
except ZeroDivisionError as e:
    logger.error(f"Caught exception: {e}")
    logger.exception("Full traceback:")
```

### Module-Specific Logging

```python
from src import get_logger

# Create a logger for this specific module
logger = get_logger("data_processor")

def process_data(data):
    logger.info(f"Processing {len(data)} records")
    # ... processing logic ...
    logger.info("Data processing completed")
```

## Integration with Existing Code

The logger has been integrated into the `backtester.py` file as an example. You can see how it replaces print statements with proper logging:

```python
# Before
print("Train prices have been previously loaded. Concatenating with test prices...")

# After
logger.info("Train prices have been previously loaded. Concatenating with test prices...")
```

## Running the Example

To see the logger in action, run the example file:

```bash
python src/logger_example.py
```

This will demonstrate all the different ways to use the logger and show the colored output.

## Best Practices

1. **Use descriptive logger names**: Use meaningful names like `get_logger("data_processor")` or `get_logger("backtester.sma")`
2. **Choose appropriate log levels**: Use DEBUG for detailed info, INFO for general progress, WARNING for potential issues, ERROR for actual errors
3. **Include context**: Add relevant information to your log messages
4. **Use exception logging**: Use `logger.exception()` to automatically include tracebacks
5. **Set log levels appropriately**: Use DEBUG during development, INFO or WARNING in production

## Configuration

The logger is configured in `src/__init__.py` with:

- Default log level: INFO
- Custom formatter with colors and timestamps
- Stream handler for console output
- Non-propagating to avoid duplicate logs

You can modify the configuration in `src/__init__.py` if you need different formatting or additional handlers (e.g., file logging). 