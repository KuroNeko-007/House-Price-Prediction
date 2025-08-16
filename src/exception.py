import sys
import traceback
from src.logger import logger

def error_message_detail(error):
    exc_type, exc_value, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return f"Error occurred in script: [{file_name}] at line [{line_no}] - {str(error)}"

class CustomException(Exception):
    def __init__(self, error):
        self.error_message = error_message_detail(error)
        super().__init__(self.error_message)

        # Optional logging
        logger.error(self.error_message)
        logger.error("Full traceback:\n" + traceback.format_exc())

    def __str__(self):
        return self.error_message
