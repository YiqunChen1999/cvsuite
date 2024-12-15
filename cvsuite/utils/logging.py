
import os
import sys
import atexit
import logging
import logging.config
from datetime import datetime
from logging import StreamHandler, FileHandler, Formatter, LoggerAdapter
from functools import lru_cache
from contextlib import contextmanager
from typing import Any, Callable

import iopath
from cvsuite.utils import comm


PROMPT = ("[%(asctime)s] [%(levelname)s] "
          "[%(name)s:%(lineno)s:%(funcName)s] ")
MESSAGE = "%(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {"format": PROMPT + MESSAGE, "datefmt": DATEFMT},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOGGING_CONFIG)


class ColorFormatter(Formatter):
    BOLD = '\033[1m'
    COLOR = '\033[1;%dm'
    RESET = "\033[0m"
    GRAY, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = list(
        map(lambda x: '\033[1;%dm' % (30 + x), range(8))
    )

    FORMATS = {
        logging.DEBUG: BLUE + PROMPT + RESET + MESSAGE,
        logging.INFO: GREEN + PROMPT + RESET + MESSAGE,
        logging.WARNING: YELLOW + PROMPT + RESET + MESSAGE,
        logging.ERROR: RED + PROMPT + RESET + MESSAGE,
        logging.CRITICAL: BOLD + RED + PROMPT + RESET + MESSAGE,
    }

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name", "") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(ColorFormatter, self).__init__(*args, **kwargs)

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = __name__.split(".")[0]
        self.logger = LoggerAdapter(logging.getLogger(name), extra={})
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.log = self.logger.log
        self.setLevel = self.logger.setLevel
        self.exception = self.logger.exception
        self.logger.setLevel(
            logging.INFO if self.is_rank_zero else logging.ERROR
        )
        setup_format()

    @lru_cache(None)
    def warning_once(self, *args, **kwargs):
        self.warning(*args, **kwargs)

    @lru_cache(None)
    def info_once(self, *args, **kwargs):
        self.info(*args, **kwargs)

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return comm.is_main_process

    @property
    def is_rank_zero(self) -> bool:
        return comm.is_main_process()

    @property
    def rank(self) -> int:
        return comm.get_rank()

    @property
    def world_size(self) -> int:
        return comm.get_world_size()


def setup_file_handler(path: str):
    root = logging.getLogger()
    handler = FileHandler(path)
    handler.setFormatter(Formatter(PROMPT + MESSAGE))
    root.addHandler(handler)
    setup_format()


def setup_format(formatter: Formatter | None = None):
    setup_libs_format()
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, FileHandler):
            continue
        elif isinstance(handler, StreamHandler):
            if formatter is None:
                formatter = ColorFormatter(PROMPT + MESSAGE)
            handler.setFormatter(formatter)


# so that calling setup_logger multiple times won't add many handlers
@lru_cache()
def setup_detectron2_logger(
        output: str | None = None,
        distributed_rank: int = 0,
        *,
        color: bool = True,
        name: str = "detectron2",
        abbrev_name: str | None = None,
        enable_propagation: bool = False,
        configure_stdout: bool = True):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        output: a file name or a directory to save log. If None, will
            not save log file. If ends with ".txt" or ".log", assumed to be
            a file name. Otherwise, logs will be saved to `output/log.txt`.
        name: the root module name of this logger
        abbrev_name: an abbreviation of the module, to avoid long names in
            logs. Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
        enable_propagation: whether to propagate logs to the parent logger.
        configure_stdout: whether to configure logging to stdout.


    Returns:
        logging.Logger: a logger
    """
    Logger(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = enable_propagation

    if abbrev_name is None:
        abbrev_name = "d2" if name == "detectron2" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if configure_stdout and distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = ColorFormatter(
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        from iopath.common.file_io import PathManager as PathManagerBase

        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager = PathManagerBase()
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))  # type: ignore  # noqa
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def create_logger(name: str | None = None,
                  save_root: str | None = None,
                  file_name: str | None = None,
                  auto_setup_fmt: bool = False,
                  verbose: bool = False):
    if name is None:
        name = __name__.split(".")[0]
    logger = Logger(name)
    if save_root is not None:
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        if file_name is None:
            curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_name = f"{curr_time}.log"
        save_path = os.path.join(save_root, file_name)
        logger.info(f"Logger messages will be saved to {save_path}")
        setup_file_handler(save_path)
    if verbose:
        logger.info(f"Logger {name} is created.")
    if auto_setup_fmt:
        setup_format()
    return logger


def setup_libs_format():
    try:
        from transformers.utils.logging import _get_library_root_logger  # type: ignore # noqa
    except ImportError:
        return
    logger = _get_library_root_logger()
    logger.propagate = True


@contextmanager
def disable_handlers(logger_name: str | None = None,
                     handler_types: tuple[logging.Handler] | None = None):
    logger = logging.getLogger(logger_name)
    if handler_types is None:
        handler_types = tuple()
    # Store the original states and disable specified handlers
    handler_levels: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if isinstance(handler, handler_types):  # type: ignore
            handler_levels.append((handler, handler.level))
            # Set to a level higher than CRITICAL
            handler.setLevel(logging.CRITICAL + 1)

    try:
        yield  # This is where the wrapped code will execute
    finally:
        # Restore the original states
        for handler, level in handler_levels:
            handler.setLevel(level)


@contextmanager
def disable_console_logging(logger_name: str | None = None):
    logger = logging.getLogger(logger_name)
    # Store the original states and disable console handlers
    handler_levels: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream in (sys.stdout, sys.stderr)):
            handler_levels.append((handler, handler.level))
            # Set to a level higher than CRITICAL
            handler.setLevel(logging.CRITICAL + 1)

    try:
        yield  # This is where the wrapped code will execute
    finally:
        # Restore the original states
        for handler, level in handler_levels:
            handler.setLevel(level)


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@lru_cache(maxsize=None)
def _cached_log_stream(filename: str):
    # use 1K buffer if writing to cloud storage
    from iopath.common.file_io import PathManager as PathManagerBase
    PathManager = PathManagerBase()
    io = PathManager.open(filename, "a", buffering=-1)
    atexit.register(io.close)
    return io
