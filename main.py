"""Entry point for the inference server."""

import uvicorn

from src.inference.app import create_app
from src.inference.config import Settings


def main() -> None:
    """Start the inference server."""
    settings = Settings()
    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        # WebSocket ping settings - disabled by default for long-running inference
        ws_ping_interval=settings.websocket_ping_interval,
        ws_ping_timeout=settings.websocket_ping_timeout,
    )


if __name__ == "__main__":
    main()
