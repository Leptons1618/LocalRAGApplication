import logging
from config import settings

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level_value)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger