from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ipa.data_type import ApiError, ApiResponse
from starlette.exceptions import HTTPException


def add_api_middleware(app: FastAPI, **kwargs):
    """Add API middleware to the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.
    """

    def _exception_handler(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            content = ApiResponse.negative(
                ApiError(code=exc.status_code, message=exc.detail)
            )
        else:
            content = ApiResponse.negative(ApiError(code=500, message=str(exc)))

        return JSONResponse(content=content.model_dump(exclude_none=True))

    app.add_exception_handler(Exception, _exception_handler)


def add_api_limiter_middleware(app: FastAPI, **kwargs):
    """
    docs: `https://slowapi.readthedocs.io/en/latest/`
    """
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(
        key_func=get_remote_address,
        **kwargs,
    )
    app.state.limiter = limiter
    # TODO: 自定义异常处理，使得返回格式统一
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
