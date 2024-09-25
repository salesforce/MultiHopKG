import logging
import os

class ColoredFormatter(logging.Formatter):
  def __init__(self, fmt, datefmt=None):
    super().__init__(fmt, datefmt)
    self.colors = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[41m'  # Red background
    }

  def format(self, record: logging.LogRecord) -> str:
    color = self.colors.get(record.levelname, '\033[0m')
    # Format the prefix including date-time and log level
    prefix = f"{self.formatTime(record, self.datefmt)} {record.levelname}"
    colored_prefix = f"{color}{prefix}\033[0m"
    message = super().format(record)
    find_record = message.find(record.levelname)
    left_over = message[find_record+len(record.levelname):]
    message = colored_prefix + left_over

    # Replace the entire prefix with the colored version
    return message

def setup_logger(
    logger_name: str, logging_level=logging.INFO, overwrite=True
):
    """
    Helper function for setting up logger both in stdout and file
    """
    # Measures necessary to take for us to be able to do multiprocess logging
    localpid = os.getpid()
    logger_name_local = f"pid({localpid})" + logger_name

    # logger_name = "SUPADEBUG"
    logger = logging.getLogger(logger_name_local)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)

    # create file handler which logs remaining debug messages
    current_cwd = os.getcwd()
    log_dir = os.path.join(
        current_cwd,
        "logs/",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{logger_name_local}.log")
    mode = "w" if overwrite else "a"
    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

