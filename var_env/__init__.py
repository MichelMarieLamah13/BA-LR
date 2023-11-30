# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import logging

PATH_LOGS = 'logs'


# Sample implementation in the 'env' module


def logging_config(log_file_path):
    """
    Configure logging settings.

    Args:
        log_file_path (str): The path to the log file.
    """
    # Create a logger
    logger = logging.getLogger()

    # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the log level
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

# Example usage:
# env.logging_config("logs/logFile_contribution_BA")
