from typing import List, Union, Any
import logging
import datetime
import os
import json
import time
from logging.handlers import TimedRotatingFileHandler
import traceback
import uuid
import pandas as pd


class Logger(logging.Logger):

    def __init__(self,
                 log_folder: str = 'logs',
                 name=None,
                 level=logging.INFO):
        super().__init__(name or 'logger', level)
        datetime_now = datetime.datetime.now().strftime("%Y-%m-%d")
        id = uuid.uuid4()
        try:
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            log_filename = os.path.join(
                log_folder,
                f'{name}_{datetime_now}_{id}.log'
            )
            file_handler = TimedRotatingFileHandler(
                log_filename,
                when="midnight",
                interval=1,
                backupCount=7
            )
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            self.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            self.addHandler(console_handler)
        except Exception as e:
            tbck_str = traceback.format_exc()
            print(f'Error initializing logger app: {e}\n\n{tbck_str}')

    def info(self, msg: str) -> None:
        super().info(msg)

    def warning(self, msg: str) -> None:
        super().warning(msg)

    def exception(self, msg: str) -> None:
        super().exception(msg)

    def error(self, msg: str) -> None:
        super().error(msg)

    def critical(self, msg: str) -> None:
        super().critical(msg)

    def debug(self, msg: str) -> None:
        super().debug(msg)

    def log_dict(self, data: dict, level=logging.INFO, prefix="") -> None:
        for key, value in data.items():
            msg = f"{prefix}{key}: {value}"
            self.log(level, msg)

    def log_list(self, data: list, level=logging.INFO, prefix="") -> None:
        for idx, item in enumerate(data):
            msg = f"{prefix}[{idx}]: {item}"
            self.log(level, msg)

    def log_json(self, data: Any, level=logging.INFO, prefix="JSON: ") -> None:
        try:
            json_data = json.dumps(data, indent=2)
            msg = f"{prefix}\n{json_data}"
            self.log(level, msg)
        except Exception as e:
            tbck_str = traceback.format_exc()
            self.error(f"Error logging JSON: {e}\n\n{tbck_str}")

    def log_elapsed_time(self,
                         start_time: Union[float, int],
                         level=logging.INFO,
                         prefix="Elapsed time: ") -> None:
        elapsed_time = time.time() - start_time
        msg = f"{prefix}{elapsed_time} seconds"
        self.log(level, msg)

    def log_progress(self,
                     current: Union[float, int],
                     total: Union[float, int],
                     level=logging.INFO,
                     prefix="Progress: ") -> None:
        progress_percentage = (current / total) * 100
        msg = f"{prefix}{current}/{total} ({progress_percentage:.2f}%)"
        self.log(level, msg)

    def log_sql_query(self,
                      query: str,
                      level=logging.INFO,
                      prefix="SQL Query: ") -> None:
        msg = f"{prefix}{query}"
        self.log(level, msg)

    def log_list_of_dicts(self,
                          list_of_dicts: List[dict],
                          level=logging.INFO,
                          prefix="Dict list: ") -> None:
        for idx, item in enumerate(list_of_dicts):
            msg = f"{prefix}[{idx}]: {item}"
            self.log(level, msg)

    def log_code_snippet(self,
                         code: str,
                         level=logging.INFO,
                         prefix="Code Snippet: ") -> None:
        msg = f"{prefix}\n{code}"
        self.log(level, msg)

    def log_request_info(self,
                         method: str,
                         path: str,
                         status_code: int,
                         level=logging.INFO,
                         prefix="HTTP Request Info: ") -> None:
        msg = f"{prefix}Method: {method}, " + \
              f"Path: {path}, " + \
              f"Status Code: {status_code}"
        self.log(level, msg)
            
    def log_pandas_dataframe(self,
                             df: pd.DataFrame,
                             level=logging.INFO,
                             prefix="DataFrame: ") -> None:
        msg = f"{prefix}\n{df}"
        self.log(level, msg)
    
