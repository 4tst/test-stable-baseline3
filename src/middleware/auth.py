from fastapi import FastAPI


def add_auth_middleware(app: FastAPI, **kwargs):
    """
    TODO: Add authentication middleware to the FastAPI application.

    该中间件最好只做简单的检查实现early fail，真正的授权验证使用官方推荐的依赖注入的方式

    Args:
        app (FastAPI): The FastAPI application instance.
    """
