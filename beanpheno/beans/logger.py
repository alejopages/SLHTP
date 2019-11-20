import logging
# import coloredlogs
import os

_FORMAT = "%(asctime)s %(levelname)s | %(message)s"
_LOGGER = None

def get_logger(level = logging.ERROR, refresh = False):
    global _LOGGER

    logging.basicConfig()

    if not _LOGGER or refresh:
        logger = logging.getLogger(__name__)
        logger.setLevel(level)
        logger.propagate = False

#        os.environ['COLOREDLOGS_LOG_FORMAT'] = _FORMAT
#        os.environ['COLOREDLOGS_LEVEL_STYLES'] = \
#            'info=20;debug=15;warning=30;error=40;critical=50'

        if not logger.handlers:
            formatter = logging.Formatter(_FORMAT, "%H:%M:%S")
            handler   = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        _LOGGER = logger

#    coloredlogs.install(logger=_LOGGER)
    return _LOGGER
