from __future__ import annotations


def main() -> None:
    """Run one state-owning HTTP worker for horizontal container scaling."""

    import uvicorn

    from boxmot.engine.service.app import app
    from boxmot.engine.service.config import ServiceSettings

    settings = ServiceSettings.from_env()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        workers=1,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
