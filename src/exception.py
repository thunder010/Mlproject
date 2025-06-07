import sys
import logging
from src.logger import logging  # Importing the logging configuration from logger module for consistent logging setup and to use the same logging format and file path

def error_message_detail(error, error_detail: sys):   # Function to extract detailed error message
    _, _, exc_tb = error_detail.exc_info()           # Extracting the traceback information
    file_name = exc_tb.tb_frame.f_code.co_filename    # Get the file name where the error occurred
    line_number = exc_tb.tb_lineno                 # Get the line number where the error occurred
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error)}]"
    return error_message

class CustomException(Exception):  # Custom exception class
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail= error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1 / 0  # Example to raise an exception
    except Exception as e:
        logging.info("Division by zero error occurred")
        raise CustomException(e, sys)