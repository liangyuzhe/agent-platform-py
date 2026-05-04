"""应用入口。"""

import logging
import logging.config
import uvicorn

# Uvicorn log config extended with application loggers
_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "fmt": "%(levelname)s:\t%(name)s\t%(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO"},
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "langsmith": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}


def main():
    uvicorn.run(
        "agents.api.app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        log_config=_LOG_CONFIG,
        reload=True,
    )


if __name__ == "__main__":
    main()
