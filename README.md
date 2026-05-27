# multi-vlm-image-labeler

多模型图像标注流水线，当前重点支持：

- 多模型并发调用
- 统一 JSON 归一化
- `caption_structured`
- 表格 / 图表结构化输出
- flowchart Mermaid graph fusion
- `accepted / review / failed` 判定

## 当前目录

```text
.
├── configs/
│   ├── models.yaml
│   ├── models.official.yaml
│   ├── models.aggregator.yaml
│   └── prompts.yaml
├── data/
├── src/
│   ├── main.py
│   ├── schema.py
│   ├── normalizer.py
│   ├── scorer.py
│   ├── decision.py
│   ├── writer.py
│   ├── graph_fusion.py
│   ├── model_clients/
│   └── validators/
├── tests/
│   └── test_graph_fusion.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

说明：

- `data/` 放输入图片
- `outputs/` 是运行产物，不纳入仓库
- `configs/` 放模型与 prompt 配置
- `src/` 放主流水线和各模块实现
- `tests/` 目前保留最小 graph fusion 回归测试

## 本地运行

推荐使用 `uv`：

```bash
uv sync --extra api --extra dev
uv run python -m src.main --data-dir data --output-dir outputs --mock --limit 5
```

如果只需要 mock：

```bash
uv sync --extra dev
uv run python -m src.main --data-dir data --output-dir outputs --mock --limit 5
```

## Docker

构建镜像：

```bash
docker build -t multi-vlm-image-labeler .
```

运行 mock：

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  multi-vlm-image-labeler \
  uv run python -m src.main \
    --data-dir /app/data \
    --output-dir /app/outputs \
    --models-config configs/models.yaml \
    --prompts-config configs/prompts.yaml \
    --mock \
    --limit 5
```

运行官方 API：

```bash
docker run --rm \
  -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
  -e ARK_API_KEY="$ARK_API_KEY" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  multi-vlm-image-labeler \
  uv run python -m src.main \
    --data-dir /app/data \
    --output-dir /app/outputs \
    --models-config configs/models.official.yaml \
    --prompts-config configs/prompts.yaml \
    --concurrent-models 3\
    --retry 1
```

## 输入与输出

支持输入格式：

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.bmp`

如果 `data/` 没有图片，会正常输出：

```text
No images found in data directory
```

运行后会生成：

- `outputs/per_image/{image_id}.json`
- `outputs/summary.jsonl`

其中：

- `per_image` 保存原始模型输出、归一化结果、验证结果、graph fusion 和最终标签
- `summary.jsonl` 保存筛选字段，适合后续人工 review

## 配置文件

- `configs/models.yaml`
  默认本地/mock 配置
- `configs/models.official.yaml`
  官方 OpenAI-compatible API 配置
- `configs/models.aggregator.yaml`
  聚合平台 OpenAI-compatible 配置
- `configs/prompts.yaml`
  标注 prompt

## 并发参数

支持的主要 CLI 参数：

- `--concurrent-models`
  同一张图最多同时请求多少个模型，默认 `3`
- `--concurrent-images`
  同时处理多少张图片，默认 `1`
- `--request-timeout`
  CLI 级默认请求超时，默认 `120`
- `--retry`
  单模型失败后的重试次数，默认 `0`

建议：

- 官方 API 从 `--concurrent-models 2` 或 `3` 起步
- 如果出现 `429` 或 timeout，先降并发，再增 `--retry`

## Graph Fusion

当多数 `image_type` 为 `flowchart` 且至少两个模型输出 Mermaid 时，系统会尝试：

- 解析多模型 Mermaid
- 节点 / 边 claim-level voting
- 融合为 fused Mermaid
- 写入 `final_label.structured_label`

额外字段：

- `final_label.structured_label.source = fused_graph`
- `final_label.structured_label.graph_confidence`
- `graph_fusion.enabled`
- `graph_fusion.node_vote_details`
- `graph_fusion.edge_vote_details`
- `graph_fusion.warnings`
- `graph_fusion.critical_errors`

flowchart 样本如果 graph fusion 风险较高，会进入 `review`。

## 测试

运行最小回归测试：

```bash
uv run python -m unittest -q tests/test_graph_fusion.py
```

## 现状说明

- `openai_compatible` client 是当前主要可用的真实 API 路径
- `mock` client 用于本地联调和结构回归
- `openai / anthropic / gemini` 目前仍是占位 client，不建议用于正式实验
- flowchart graph fusion 仍是启发式实现，复杂重复节点场景会倾向进入 `review`
