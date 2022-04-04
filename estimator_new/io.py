# -*- coding: utf-8 -*-
import logging


class Logging:
    """
    Control level of detail being printed.
    """

    plain_logger = logging.StreamHandler()
    plain_logger.setFormatter(logging.Formatter("%(message)s"))

    detail_logger = logging.StreamHandler()
    detail_logger.setFormatter(logging.Formatter("[%(name)8s] %(message)s"))

    logging.getLogger("estimator").handlers = [plain_logger]
    logging.getLogger("estimator").setLevel(logging.INFO)

    loggers = ("batch", "bdd", "usvp", "bkw", "gb", "repeat", "guess", "bins", "dual")

    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    LEVEL0 = logging.INFO
    LEVEL1 = logging.INFO - 2
    LEVEL2 = logging.INFO - 4
    LEVEL3 = logging.INFO - 6
    LEVEL4 = logging.INFO - 8
    LEVEL5 = logging.DEBUG
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    for logger in loggers:
        logging.getLogger(logger).handlers = [detail_logger]
        logging.getLogger(logger).setLevel(logging.INFO)

    @staticmethod
    def set_level(lvl, loggers=None):
        """Set logging level

        :param lvl: one of `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `LEVELX`, `DEBUG`, `NOTSET` with `X` ∈ [0,5]
        :param loggers: one of `Logging.loggers`, if `None` all loggers are used.

        """
        if loggers is None:
            loggers = Logging.loggers

        for logger in loggers:
            logging.getLogger(logger).setLevel(lvl)

    @classmethod
    def log(cls, logger, level, msg, *args, **kwds):
        level = int(level)
        return logging.getLogger(logger).log(
            cls.INFO - 2 * level, f"{{{level}}} " + msg, *args, **kwds
        )
