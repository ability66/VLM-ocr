FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

COPY requirements.txt pyproject.toml uv.lock ./

RUN uv pip install --system torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --system -r requirements.txt

COPY configs ./configs
COPY scripts ./scripts
COPY data/flowchart_crops ./data/flowchart_crops
COPY README.md ./

RUN mkdir -p \
    /app/data/pdfs \
    /app/data/page_images \
    /app/data/flowchart_crops/manual_mermaid \
    /app/outputs/flowchart_graph/raw \
    /app/outputs/flowchart_graph/normalized \
    /app/outputs/flowchart_graph/review \
    /app/outputs/flowchart_graph/reports \
    && python -m compileall scripts

CMD ["bash"]
