from fastapi import FastAPI


def add_pagination_middleware(app: FastAPI, **kwargs):
    """
    docs: `https://uriyyo-fastapi-pagination.netlify.app/`
    """
    from fastapi_pagination import add_pagination

    add_pagination(app, **kwargs)
