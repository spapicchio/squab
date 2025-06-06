import logging
import os


def get_logger(name: str, level=logging.INFO, log_file: str = None):
    """
    Create and return a customized logger with error message coloring.
    
    Args:
        name (str): Name of the logger.
        level (int): Logging level, default is logging.INFO.
        log_file (str): File path where logs should be written.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler with color support for ERROR
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Use the custom formatter that highlights ERROR messages
        plain_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
        console_handler.setFormatter(plain_formatter)
        logger.addHandler(console_handler)
        # File handler (logs to file in plain text, without colors)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(plain_formatter)
            logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger("test_logger", level=logging.DEBUG, log_file="./test_logger.log")
    logger.error("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
