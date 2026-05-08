"""应用入口。"""

import logging
import logging.config
import os
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
    reload_enabled = os.getenv("UVICORN_RELOAD", "").lower() in {"1", "true", "yes", "on"}
    uvicorn.run(
        "agents.api.app:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8080")),
        log_level="info",
        log_config=_LOG_CONFIG,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
