"""应用入口。"""

import logging
import uvicorn


def main():
    # Ensure langsmith trace logs are visible
    logging.getLogger("langsmith").setLevel(logging.INFO)

    uvicorn.run(
        "agents.api.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )


if __name__ == "__main__":
    main()
