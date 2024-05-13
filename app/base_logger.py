"""
Module to config base logger shared by all modules.
"""
import logging
from datetime import datetime

logger = logging
log_file = f"log/{datetime.now().strftime('%B-%d-%y')}.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s]:%(name)s: %(message)s"

logger.basicConfig(format=LOG_FORMAT,
                   level=logging.INFO,
                   filename=log_file,
                   datefmt='%I:%M:%S%p')
