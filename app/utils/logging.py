import logging
import sys

COLOR_RESET  = "\033[0m"
COLOR_BOLD   = "\033[1m"
COLOR_RED    = "\033[31m"
COLOR_GREEN  = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN   = "\033[36m"
COLOR_GRAY   = "\033[90m"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    COLOR_GRAY,
        logging.INFO:     COLOR_GREEN,
        logging.WARNING:  COLOR_YELLOW,
        logging.ERROR:    COLOR_RED,
        logging.CRITICAL: COLOR_RED + COLOR_BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, COLOR_RESET)
        fmt = (
            f"{COLOR_GRAY}%(asctime)s{COLOR_RESET} | "
            f"{color}%(levelname)-8s{COLOR_RESET} | "
            f"{COLOR_CYAN}%(name)s{COLOR_RESET} | "
            f"%(message)s"
        )
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with coloured console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    return logger
