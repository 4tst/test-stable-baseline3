__doc__ = """
Middleware module for the FastAPI application.
"""

import logging
import re

from fastapi import FastAPI

from .api import add_api_limiter_middleware, add_api_middleware
from .auth import add_auth_middleware
from .cors import add_cors_middleware
from .log import add_log_middleware
from .pagination import add_pagination_middleware


def add_all_middleware(app: FastAPI, strict: bool = True, **kwargs):
    """Add all middleware to the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.
        strict (bool, optional): Whether to raise an exception if a middleware fails. Defaults to True.
        kwargs (dict, optional): Additional keyword arguments to pass to each middleware function. Defaults to {}.
    """
    for f in [
        add_api_middleware,
        add_api_limiter_middleware,
        add_auth_middleware,
        add_cors_middleware,
        add_log_middleware,
        add_pagination_middleware,
    ]:
        match = re.match(r"^add_([a-z_]+)_middleware$", f.__name__)
        if not match:
            raise ValueError(
                f"Invalid middleware function name: {f.__name__}, expected format: `add_[your_middleware_name]_middleware`"
            )
        name = match.group(1)
        try:
            f(app, **(kwargs.get(name, {})))
        except Exception as e:
            if strict:
                raise e
            logging.error(f"Error adding middleware {f.__name__}: {e}")
