import logging
from pathlib import Path


def get_logger(log_path: Path, name: str = None, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)

    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = False

    return logger
