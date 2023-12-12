# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import logging

PATH_LOGS = 'logs'


def logging_config(log_file_path):
    """
    Configure logging settings.

    Args:
        log_file_path (str): The path to the log file.
    """
    logging.basicConfig(
        filename=log_file_path,  # Log file name
        level=logging.DEBUG,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
    )

# Example usage:
