import sys
from pathlib import Path
from typing import Union

from alembic import command
from alembic.config import Config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent

DEBUG = True

def get_runtime_root_dir():
    """
    获取项目根目录。
    1. 如果是打包后的应用（如使用 PyInstaller），则返回 _MEIPASS 目录。
    2. 否则，返回本项目的根目录
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return PROJECT_ROOT_DIR


RUNTIME_ROOT_DIR = get_runtime_root_dir()
DATA_DIR = RUNTIME_ROOT_DIR / "data"
SIDE_CAR_DIR = RUNTIME_ROOT_DIR / "sidecar"
STATIC_DIR = RUNTIME_ROOT_DIR / "static"

for d in [DATA_DIR, SIDE_CAR_DIR, STATIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

APP_NAME = "fastapi-starter"

DB_PATH = DATA_DIR / f"{APP_NAME}.db"
DB_URL = f"sqlite:///{DB_PATH}"
ASYNC_DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"

def setup_alembic(config: Union[Config, None] = None):
    """
    如果alembic中配置了sqlalchemy.url，则使用该URL；
    否则，使用conf中定义的DB_URL
    """
    cfg = config or Config(RUNTIME_ROOT_DIR / "alembic.ini")
    if not cfg.get_main_option("sqlalchemy.url"):
        cfg.set_main_option("sqlalchemy.url", DB_URL)
    return cfg


def setup_database():
    """
    初始化数据库，运行数据库迁移脚本
    """

    cfg = setup_alembic()
    command.upgrade(cfg, "head")
