import logging

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

def setup_logger(logger):
    format = 'PID={process}/{processName} @ {funcName}, {filename}:{lineno} :: {asctime} {levelname}: {message}'
    logging.basicConfig(format=format, style='{')
    logger.setLevel(logging.INFO)

logger = logging.getLogger('iml')
setup_logger(logger)
