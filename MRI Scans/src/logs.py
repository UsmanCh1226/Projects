import logging
import os

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(message)s')

        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

