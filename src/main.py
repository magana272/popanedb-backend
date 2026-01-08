"""
POPANE Emotion Database API
Main application entry point

This is the root of the project which initializes the FastAPI app
and includes all module routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.constants import Environment
from src.studies.router import router as studies_router
from src.visualization.router import router as visualization_router


def create_app() -> FastAPI:
    """
    Application factory pattern for creating the FastAPI app.

    Benefits:
    - Easier testing with different configurations
    - Clear initialization flow
    - Better organization
    """
    # Configure app based on environment
    app_configs = {
        "title": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "API for accessing physiological emotion data from the POPANE database",
    }

    # Hide docs in production (unless explicitly enabled)
    SHOW_DOCS_ENVIRONMENTS = (Environment.DEVELOPMENT, Environment.STAGING)
    if settings.ENVIRONMENT not in SHOW_DOCS_ENVIRONMENTS:
        app_configs["openapi_url"] = None

    app = FastAPI(**app_configs)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # Include routers
    app.include_router(studies_router)
    app.include_router(visualization_router)

    # Health check endpoint
    @app.get("/", tags=["Health"])
    def health_check():
        """API health check endpoint"""
        return {
            "status": "ok",
            "message": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT
        }

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
