import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',  
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)
