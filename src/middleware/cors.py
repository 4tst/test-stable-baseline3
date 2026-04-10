from fastapi import FastAPI


def add_cors_middleware(app: FastAPI, **kwargs):
    """Add CORS middleware to the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        **kwargs,
    )
