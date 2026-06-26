"""FastAPI application entry point for the QueueStorm Investigator backend.

This module wires the FastAPI app, registers routers, and exposes the
``GET /health`` liveness probe required by ``AGENT.md`` §3. Business-logic
endpoints (e.g. ``POST /analyze-ticket``) are mounted under ``app/api/``.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.core.config import get_settings
from app.models.schemas import HealthResponse

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""

    settings = get_settings()

    app = FastAPI(
        title="QueueStorm Investigator API",
        description=(
            "AI/API copilot for a digital finance platform's support team. "
            "Investigates customer complaints against transaction history and "
            "returns a structured routing + drafted-reply response."
        ),
        version="0.1.0",
        contact={"name": "QueueStorm Investigator Team"},
    )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness probe required by the API contract (must return ``ok``)."""

        return HealthResponse(status="ok")

    # Routers will be mounted here as additional endpoints land:
    # from app.api.analyze import router as analyze_router
    # app.include_router(analyze_router)

    _ = settings  # reserved for future startup configuration (logging, etc.)

    return app


# Module-level ASGI app for ``uvicorn app.main:app``
app: FastAPI = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level,
    )