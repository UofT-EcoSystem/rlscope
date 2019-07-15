import logging

def setup_logging():
    format = 'PID={process}/{processName} @ {funcName}, {filename}:{lineno} :: {asctime} {levelname}: {message}'
    logging.basicConfig(format=format, style='{', level=logging.INFO)

    # Trying to disable excessive DEBUG logging messages coming from luigi,
    # but it's not working....
    luigi_logger = logging.getLogger('luigi-interface')
    luigi_logger.setLevel(logging.INFO)
    luigi_logger = logging.getLogger('luigi')
    luigi_logger.setLevel(logging.INFO)
