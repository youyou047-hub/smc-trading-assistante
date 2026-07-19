
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_file_path: str = "logs/trading_assistant.log", level: str = "INFO") -> logging.Logger:
    """Sets up a structured logger with file rotation and console output.

    Args:
        log_file_path (str): Path to the log file.
        level (str): Logging level (e.g., "INFO", "DEBUG", "WARNING").

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("trading_assistant")
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent messages from being passed to the root logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        logger.addHandler(console_handler)

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    # Example usage
    logger = setup_logger()
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    # Test with a different log file and level
    test_logger = setup_logger("logs/test.log", "DEBUG")
    test_logger.debug("This is a debug message to test.log")
