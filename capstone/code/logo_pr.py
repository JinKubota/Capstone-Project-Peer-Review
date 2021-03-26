import os
import sys
import json
import time
import re
import datetime
import pandas as pd
import numpy as np
import logging

LOG_LEVEL= "DEBUG"
LOGGING_TO_FILE= "True"
LOGGING_TO_STREAM= "True"

class OneLineExceptionFormatter(logging.Formatter):
    """ Formats exception info into a single line by removing newline """
    def formatException(self, exc_info):
        try:
            result = super(OneLineExceptionFormatter, self).formatException(exc_info)
            if result:
                result = result.replace('\n', ' | ')
            return repr(result)     # or format into one line however you want to
        except Exception:
            print("LOGGER ERROR: Couldn't format exception info with error trace - ", sys.exc_info())
            return False

    def format(self, record):
        try:
            s_var = super(OneLineExceptionFormatter, self).format(record)
            if record.exc_text:
                s_var = s_var.replace('\n', ' | ')
            return s_var
        except Exception:
            print("LOGGER ERROR: Couldn't format record info with error trace - ", sys.exc_info())
            return False



def get_logger(log_path, app_name):
    """ LOG Function """
    current_date = datetime.date.today()
    current_date_formatted = datetime.date.strftime(current_date, '%d-%m-%Y')
    logger = logging.getLogger(app_name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    # Here we define our formatter
    formatter = OneLineExceptionFormatter('%(process)d - %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not formatter:
        formatter = logging.Formatter('%(process)d - %(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # This step is to avoid duplicate logging for same logger initiation (with same app_name)
    if logger.handlers:
        logger.handlers = []

    if LOGGING_TO_FILE.upper() == "TRUE":
        try:
            try:
                os.makedirs(log_path)
                print("LOG directory/folder created for ", app_name)
            except FileExistsError:
                print("LOG directory/folder already exists for ", app_name)

            log_file_handler = logging.FileHandler(log_path + "/cap_" + str(current_date_formatted) +
                                                   ".log")
            log_file_handler.setLevel(LOG_LEVEL)
            log_file_handler.setFormatter(formatter)
            logger.addHandler(log_file_handler)            
        except Exception:
            print("LOGGER ERROR: Couldn't create LOGGER File Handler with error trace - ",
                  str(sys.exc_info()).replace("\n", " | "))

    ########################################################################
    # STREAM HANDLER
    ########################################################################
    if LOGGING_TO_FILE.upper() != "TRUE" and LOGGING_TO_STREAM.upper() != "TRUE":
        print("IMPORTANT NOTE on LOGGING : Cannot switch off both file and stream logging. "
              "Switching on default stream logging...")
        log_stream_handler = logging.StreamHandler()
        log_stream_handler.setFormatter(formatter)
        log_stream_handler.setLevel(LOG_LEVEL)
        logger.addHandler(log_stream_handler)

    if LOGGING_TO_STREAM.upper() == "TRUE":
        log_stream_handler = logging.StreamHandler()
        log_stream_handler.setFormatter(formatter)
        log_stream_handler.setLevel(LOG_LEVEL)
        logger.addHandler(log_stream_handler)

    ########################################################################

    return logger
