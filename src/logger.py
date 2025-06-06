import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # this will create a log file with the current date and time
logs_path= os.path.join(os.getcwd(), "logs", LOG_FILE)  # logs will be stored in a 'logs' directory in the current working directory
os.makedirs(logs_path, exist_ok=True)      # Create the logs directory if it doesn't exist

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)    # Full path to the log file

logging.basicConfig(                    # Basic configuration for logging and setting the log file
    filename=LOG_FILE_PATH,         # Log file path
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",   # it will log the time, line number, logger name, log level, and message
    level=logging.INFO          # Set the logging level to INFO such that all messages at this level and above will be logged for eg INFO, WARNING, ERROR, CRITICAL
)