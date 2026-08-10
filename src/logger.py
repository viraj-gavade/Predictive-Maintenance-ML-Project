from datetime import datetime
import logging
import os 
from pathlib import Path
project_root = Path(__file__).resolve().parent

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = os.path.join(project_root, 'Logs' , LOG_FILE)
os.makedirs(log_dir,exist_ok=True)


LOG_FILE_PATH = os.path.join(log_dir,LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s -%(message)s',
    level=logging.INFO
)


if __name__ == "__main__":
    logging.info('Logging started ')