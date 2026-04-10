# 固定镜像版本，保证环境稳定
FROM python:3.11.10-slim-bookworm

# 工作目录
WORKDIR /app

# ==================== 系统优化 ====================
# 禁用 Python 缓存 + 标准输出无缓冲（生产必备）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# ==================== 先安装系统依赖 ====================
# 只安装必须的包，安装后清理缓存，减小体积
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ==================== 先复制依赖文件（核心优化！） ====================
# 只复制依赖清单，依赖不变 → 永远用缓存，不重装
COPY pyproject.toml uv.lock ./

# ==================== 安装依赖（缓存层） ====================
RUN pip install --upgrade pip uv && \
    uv sync --all-extras --index-strategy unsafe-best-match --no-cache

# ==================== 最后复制代码（不影响依赖缓存） ====================
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "main.py", "serve"]