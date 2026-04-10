import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from typer import Typer

from biz import demo_biz_add
from conf import STATIC_DIR
from middleware import add_all_middleware

# TODO: i don't want to add this manually, but i have no idea how to do it automatically at present
sys.path.append(str(Path(__file__).parent / "src"))

# print(sys.path)

cmd = Typer()
app = FastAPI()
add_all_middleware(app, strict=True)


@app.get("/ping")
@app.state.limiter.limit("5/minute")
async def ping(request: Request) -> str:
    return "pong"


@app.get("/error")
async def error():
    return 1 / 0


# 这一行最好放在所有路由之后，以避免路由冲突
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.on_event("startup")
async def on_startup():
    pass


@cmd.command()
def add(a: int, b: int) -> int:
    return demo_biz_add(a, b)


@cmd.command()
def sub(a: int, b: int) -> int:
    c = a - b
    print(f"sub({a}, {b}) = {c}")
    return c


@cmd.command()
def setup_db():
    from conf import setup_database

    # if you need to use database, invoke setup_database freely
    setup_database()


@cmd.command()
def version():
    from my_sdk.version import __version__

    print(__version__)


@cmd.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    from conf import setup_database

    setup_database()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cmd()
