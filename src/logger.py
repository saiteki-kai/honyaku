import logging
import sys

from pathlib import Path


def setup_logging(logger: logging.Logger, log_file: Path) -> None:
    LOG_FORMAT = "[%(asctime)s] %(levelname)s: [%(process)s] %(name)s: %(message)s"

    if not log_file.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)

    def log_unhandled_exceptions(exc_type, exc_value, exc_traceback):
        logger.error("Unhandled exception occurred", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_unhandled_exceptions
