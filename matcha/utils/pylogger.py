import logging

from lightning.pytorch.utilities import rank_zero_only


# 初始化一个适用于多GPU环境的日志记录器，通过使用rank_zero_only装饰器，确保所有日志级别都标记为rank为0的进程
def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    :param name: The name of the logger, defaults to ``__name__``.

    :return: A logger object.
    """
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))  # rank_zero_only只允许在rank为0的进程中执行

    return logger
