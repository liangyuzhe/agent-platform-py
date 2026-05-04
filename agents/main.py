"""应用入口。"""

import logging
import uvicorn


def main():
    # Configure root logger so application logs (INFO+) are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:\t%(name)s\t%(message)s",
    )

    # Ensure langsmith trace logs are visible
    logging.getLogger("langsmith").setLevel(logging.INFO)

    uvicorn.run(
        "agents.api.app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=True,
    )


if __name__ == "__main__":
    main()
