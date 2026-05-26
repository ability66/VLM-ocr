FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests

RUN uv sync --locked --no-dev --extra api

VOLUME ["/app/data", "/app/outputs"]

CMD ["uv", "run", "python", "-m", "src.main", "--data-dir", "/app/data", "--output-dir", "/app/outputs"]
