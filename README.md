# Flowchart Graph Benchmark

项目目标只保留一条主线：

流程图图片 / 从 PDF 页面裁剪出的流程图区域  
→ 识别成 `graph JSON` 和 `Mermaid`  
→ 生成人工 review 页面  
→ 对比工具效果

当前仓库已经验证通过的自动工具是：

- `diagram2graph_hf` = `zackriya/diagram2graph`

当前样例结果：

- `figure1`: `success`, `19 nodes`, `18 edges`
- `figure2`: `success`, `21 nodes`, `19 edges`
- `figure3`: `success`, `13 nodes`, `10 edges`
- `figure4`: `success`, `10 nodes`, `8 edges`

## 推荐启动方式

如果你只是想换一台服务器快速复测，优先用 Docker：

- Docker 版：环境最稳定，适合迁移到新 GPU 服务器
- 本机 `uv` 版：适合调试代码和开发

## 已验证环境

以下是当前这台服务器上实际跑通 `diagram2graph_hf` 的环境：

- Python `3.11`
- GPU `Tesla V100-DGXS-16GB`
- `torch 2.6.0+cu124`
- `transformers 4.51.2`
- `accelerate 1.13.0`
- `Pillow 10.4.0`
- `qwen-vl-utils 0.0.14`

说明：

- V100 不支持 `bfloat16`，仓库里的 `diagram2graph_hf` adapter 已改成自动选择 dtype，当前会落到 `float16`
- `diagram2graph_hf` 首次运行会下载 Hugging Face 权重，第一次通常最慢

## 目录

```text
.
├── data/
│   ├── pdfs/
│   ├── page_images/
│   └── flowchart_crops/
├── outputs/
│   └── flowchart_graph/
│       ├── raw/
│       ├── normalized/
│       ├── review/
│       └── reports/
├── scripts/
│   ├── common.py
│   ├── flowchart_graph_common.py
│   ├── render_pages.py
│   ├── list_flowchart_crops.py
│   ├── crop_flowcharts.py
│   ├── run_flowchart_graph_tools.py
│   ├── normalize_flowchart_graph_outputs.py
│   ├── make_flowchart_graph_review.py
│   └── summarize_flowchart_graph_results.py
├── configs/
│   └── tools.yaml
├── Dockerfile
├── compose.yaml
└── pyproject.toml
```

## Docker 运行

这是推荐方式，尤其适合换服务器重测。

### 1. 宿主机前置条件

宿主机需要：

- 已安装 Docker
- 已安装 NVIDIA 驱动
- 已安装 `nvidia-container-toolkit`

确认 Docker 能看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

### 2. 构建镜像

在仓库根目录执行：

```bash
docker build -t vlm-ocr:latest .
```

镜像里会安装：

- Python `3.11`
- `torch 2.6.0 + cu124`
- `transformers 4.51.2`
- `accelerate 1.13.0`
- `qwen-vl-utils 0.0.14`

### 3. 启动容器

#### 方式 A：直接用 `docker run`

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/data/flowchart_crops:/app/data/flowchart_crops" \
  -v "$(pwd)/data/pdfs:/app/data/pdfs" \
  -v "$(pwd)/data/page_images:/app/data/page_images" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vlm-ocr:latest
```

#### 方式 B：用 `docker-compose`

```bash
docker-compose -f compose.yaml run --rm vlm-ocr
```

如果你的机器安装的是新版 Compose 插件，也可以：

```bash
docker compose -f compose.yaml run --rm vlm-ocr
```

### 4. 容器内单图 smoke test

进入容器后执行：

```bash
python scripts/run_flowchart_graph_tools.py --tools diagram2graph_hf --crop-id figure1
```

### 5. 容器内跑全部 crop

```bash
python scripts/list_flowchart_crops.py
python scripts/run_flowchart_graph_tools.py --tools diagram2graph_hf
python scripts/normalize_flowchart_graph_outputs.py --tools diagram2graph_hf
python scripts/make_flowchart_graph_review.py
python scripts/summarize_flowchart_graph_results.py
```

### 6. 宿主机查看结果

结果都在宿主机挂载目录里：

- `outputs/flowchart_graph/raw/`
- `outputs/flowchart_graph/normalized/`
- `outputs/flowchart_graph/review/index.html`
- `outputs/flowchart_graph/reports/flowchart_run_summary.csv`

### 7. Docker 方式的注意点

- 第一次运行 `diagram2graph_hf` 会下载 Hugging Face 模型，第一次最慢
- 模型缓存通过挂载 `$HOME/.cache/huggingface` 复用，后续运行会快很多
- 当前镜像默认面向 GPU 推理，不是 CPU-only 镜像
- 如果宿主机 CUDA 环境和 `cu124` 明显不兼容，需要改 Dockerfile 里的 `torch` 安装命令

## 从零开始配环境

下面这些步骤是给一台新服务器准备的，按顺序执行即可。

### 1. 安装 `uv`

如果机器上还没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

装完后重新打开 shell，或者手动把 `uv` 加进 `PATH`。

确认：

```bash
uv --version
```

### 2. 创建项目目录并进入仓库

如果是你自己从 GitHub 拉下来的仓库：

```bash
git clone <your-repo-url>
cd ocr
```

当前仓库地址：

```bash
git clone git@github.com:ability66/VLM-ocr.git
cd VLM-ocr
```

### 3. 创建虚拟环境

```bash
uv venv .venv --python 3.11
```

### 4. 安装 PyTorch

这一步不要直接照抄所有机器都一样的命令，应该按你服务器的 CUDA 环境来装。

当前这台已验证服务器使用的是：

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

如果你的服务器 CUDA 版本不同，改用对应的官方 PyTorch 安装命令。

确认 GPU 是否可用：

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

### 5. 安装项目依赖

```bash
uv pip install -r requirements.txt
```

这些依赖里已经包含：

- `transformers`
- `accelerate`
- `qwen-vl-utils`
- `pymupdf`
- `pillow`

### 6. 可选：确认关键依赖版本

```bash
uv run python - <<'PY'
import importlib
mods = ['torch', 'transformers', 'accelerate', 'PIL', 'qwen_vl_utils']
for name in mods:
    m = importlib.import_module(name)
    print(name, getattr(m, '__version__', 'OK'))
PY
```

### 7. 编译检查脚本

```bash
uv run python -m compileall scripts
```

## 输入数据准备

### 方式 A：直接放流程图 crop

把你手工截图 / 裁剪好的流程图图片放到：

```text
data/flowchart_crops/
```

支持：

- `.png`
- `.jpg`
- `.jpeg`

### 方式 B：先从 PDF 渲染整页，再按 bbox 裁剪

先把 PDF 放到：

```text
data/pdfs/
```

渲染整页图片：

```bash
uv run python scripts/render_pages.py
```

如果你已经有 bbox，再用：

```bash
uv run python scripts/crop_flowcharts.py
```

默认会读取：

```text
outputs/flowchart_graph/reports/flowchart_crop_template.csv
```

## 主流程

### Step 1. 扫描输入 crop

```bash
uv run python scripts/list_flowchart_crops.py
```

输出：

```text
outputs/flowchart_graph/reports/flowchart_inputs.csv
```

如果 `data/flowchart_crops/` 为空，会提示：

```text
请先放入流程图 crop 图片。
```

### Step 2. 单图 smoke test

先只跑一张，确认服务器环境正常：

```bash
uv run python scripts/run_flowchart_graph_tools.py --tools diagram2graph_hf --crop-id figure1
```

如果你要先自动取第一张图的名字，可以自己看：

```bash
ls data/flowchart_crops
```

这一步成功后，输出会落到：

```text
outputs/flowchart_graph/raw/diagram2graph_hf/figure1/
```

重点文件：

- `run_status.json`
- `raw_output.txt`
- `raw_output.json`
- `nodes.json`
- `edges.json`
- `mermaid.mmd`
- `parsed_graph.json`

### Step 3. 跑全部 crop

```bash
uv run python scripts/run_flowchart_graph_tools.py --tools diagram2graph_hf
```

当前仓库还保留了这些 adapter 配置：

- `flowchart2mermaid_manual`
- `diagram2graph_hf`
- `mineru_pro_image_analysis`
- `vlm_json_mermaid`
- `flowextract_external`

但自动识图优先看：

- `diagram2graph_hf`

### Step 4. 标准化输出

```bash
uv run python scripts/normalize_flowchart_graph_outputs.py --tools diagram2graph_hf
```

输出目录：

```text
outputs/flowchart_graph/normalized/
```

统一 schema：

```json
{
  "crop_id": "",
  "tool": "",
  "nodes": [],
  "edges": [],
  "mermaid": "",
  "raw_output": "",
  "status": "success|failed|skipped|partial",
  "parse_errors": []
}
```

### Step 5. 生成 review 页面

```bash
uv run python scripts/make_flowchart_graph_review.py
```

输出：

- `outputs/flowchart_graph/review/index.html`
- `outputs/flowchart_graph/reports/flowchart_manual_scores.csv`

### Step 6. 整理结果汇总

```bash
uv run python scripts/summarize_flowchart_graph_results.py
```

输出：

- `outputs/flowchart_graph/reports/flowchart_run_summary.csv`
- `outputs/flowchart_graph/reports/flowchart_run_summary.md`

### Step 7. 打开页面查看

打开：

```text
outputs/flowchart_graph/review/index.html
```

页面左侧是原始 crop，右侧会展示：

- `status`
- `nodes`
- `edges`
- `mermaid`
- `raw output`

## 当前默认配置

配置文件在：

- `configs/tools.yaml`

当前推荐直接使用：

- `diagram2graph_hf`

当前这台机器上已经验证通过的关键配置是：

- `torch_dtype = float16`
- `max_new_tokens = 4096`
- `retry_max_new_tokens = 6144`

## 换服务器时最容易出的问题

### 1. GPU 不支持 `bfloat16`

V100 就不支持，所以必须用：

- `float16`
- 或 `torch_dtype="auto"`

仓库当前代码已经处理好了。

### 2. 首次下载模型很慢

`zackriya/diagram2graph` 第一次运行会从 Hugging Face 拉权重，可能十几分钟。

### 3. 模型输出 JSON 被截断

仓库当前已经把：

- `max_new_tokens = 4096`
- `retry_max_new_tokens = 6144`

如果还是截断，可以继续加大。

### 4. review 页面显示旧结果

重新按顺序跑：

```bash
uv run python scripts/normalize_flowchart_graph_outputs.py --tools diagram2graph_hf
uv run python scripts/make_flowchart_graph_review.py
```

如果浏览器还显示旧内容，强刷页面。
