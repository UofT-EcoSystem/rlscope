"""
RL-Scope log message configuration.

RL-Scope uses different colours to mark different log messages:

* INFO: green
* WARNING: yellow
* ERROR: red
* DEBUG: white

DEBUG messages are enabled by default, and must be disabled using
:py:func:`rlscope.profiler.rlscope_logging.disable_debug_logging`.

Examples
--------
>>> from rlscope.profiler.rlscope_logging import logger
... from rlscope.profiler import rlscope_logging
...
... # e.g., print a informational message in green
... logger.info("This info message is seen")
...
... # e.g., disable rlscope debug messages
... rlscope_logging.disable_debug_logging()
... logger.debug("This debug message is NOT seen")
...
... # e.g., print an error message in red
... logger.error("This is an error")

Attributes
----------
logger : logging.Logger
    Singleton Logger object used throughout rlscope module for logging coloured info/debug/errors messages with line numbers.
"""
import sys
import logging
import re

import colorlog
import progressbar

# def setup_logging():
#     format = 'PID={process}/{processName} @ {funcName}, {filename}:{lineno} :: {asctime} {levelname}: {message}'
#     logger.basicConfig(format=format, style='{', level=logging.INFO)
#
#     # Trying to disable excessive DEBUG logging messages coming from luigi,
#     # but it's not working....
#     luigi_logger = logger.getLogger('luigi-interface')
#     luigi_logger.setLevel(logging.INFO)
#     luigi_logger = logger.getLogger('luigi')
#     luigi_logger.setLevel(logging.INFO)

def get_logger():
    logger = logging.getLogger('rlscope')
    return logger

_handler = None
logger = get_logger()
def setup_logger(debug=False, colors=None, line_numbers=None):
    global _handler, logger

    # NOTE: cannot import at module level since py_config imports this module.
    from rlscope import py_config

    # If we installed with a wheel file (a.k.a., production mode), don't show line number information.
    # If we installed with "python setup.py" (a.k.a., development mode), show line number information.
    if line_numbers is None:
        line_numbers = debug or py_config.is_development_mode()

    # logger = get_logger()

    # Weird interaction with absl.logging package where log messages print twice...
    # I guess absl registers a stream handler; lets just NOT propagate our messages to the root logger then.
    # NOTE: This might break another module from capturing our RL-Scope logs to file...oh well.
    # https://docs.python.org/3/library/logging.html#logging.Logger.propagate
    logger.propagate = False

    def _remove_colors(logformat):
        logformat = re.sub(re.escape('%(log_color)s'), '', logformat)
        logformat = re.sub(re.escape('%(reset)s'), '', logformat)
        return logformat

    if colors is None:
        if sys.stdout.isatty():
            # You're running in a real terminal.
            # Use colors.
            colors = True
        else:
            # User is pipeing command to something.
            # Don't user colors.
            colors = False

    # Setting up colored logs.
    # https://stackoverflow.com/a/23964880

    def _append_logformat(logformat, txt):
        if logformat == "":
            return txt
        return logformat + " " + txt

    logformat = ""
    # "INFO   |"
    logformat = _append_logformat(logformat, "%(log_color)s%(levelname)-7s%(reset)s |")
    if line_numbers:
        # Show details about where in the source error happened (e.g., process/pid, file, line number, function name)
        # "PID=130647/MainProcess @ do_plot, calibration.py:1008 2021-02-01 20:21:26,320"
        logformat = _append_logformat(logformat, "PID=%(process)s/%(processName)s @ %(funcName)s, %(filename)s:%(lineno)s %(asctime)s")
    # <Colour-coded message.>
    logformat = _append_logformat(logformat, "%(log_color)s%(message)s%(reset)s")

    if not colors:
        logformat = _remove_colors(logformat)

    if debug:
        # This shows DEBUG messages.
        loglevel = logging.DEBUG
    else:
        # Hide DEBUG message (still shows INFO, WARNING, ERROR, CRITICAL)
        loglevel = logging.INFO

    # This hides DEBUG messages.
    # loglevel = logging.INFO

    # Only WARNING and ERROR (hides INFO)
    # loglevel = logging.WARNING

    # Only ERROR and CRITICAL (hides ERROR)
    # loglevel = logging.ERROR

    progressbar.streams.wrap_stderr()

    logging.root.setLevel(loglevel)
    formatter = colorlog.ColoredFormatter(
        logformat,
        # NOTE: ColoredFormatter will add a reset character even if it isn't needed...
        # so lets just do it ourselves by adding "%(reset)s" to the end of our logformat above.
        reset=False
    )
    new_handler = logging.StreamHandler()
    new_handler.setLevel(loglevel)
    new_handler.setFormatter(formatter)
    logger.setLevel(loglevel)
    logger.addHandler(new_handler)
    if _handler is not None:
        # Remove old logging handler.
        logger.removeHandler(_handler)
    _handler = new_handler

# Default setup: debug mode with colors and line numbers
setup_logger(debug=False, colors=True)

def enable_debug_logging(line_numbers=None):
    setup_logger(debug=True, colors=True, line_numbers=line_numbers)

def disable_debug_logging(line_numbers=None):
    setup_logger(debug=False, colors=True, line_numbers=line_numbers)
