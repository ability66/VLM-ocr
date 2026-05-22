from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from common import PROJECT_ROOT, console, ensure_parent, load_yaml, read_csv_rows, read_json, relative_path, write_csv
from flowchart_graph_common import (
    FLOWCHART_CROPS_DIR,
    FLOWCHART_NORMALIZED_DIR,
    FLOWCHART_REPORTS_DIR,
    FLOWCHART_REVIEW_DIR,
    build_crop_records,
    ensure_flowchart_dirs,
    image_data_uri,
)


SCORE_FIELDS = [
    "crop_id",
    "tool",
    "node_score",
    "edge_score",
    "direction_score",
    "label_score",
    "mermaid_valid",
    "overall_score",
    "major_errors",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HTML review for flowchart graph benchmark outputs.")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/flowchart_crops"), help="Directory containing flowchart crop images.")
    parser.add_argument("--normalized-dir", type=Path, default=Path("outputs/flowchart_graph/normalized"), help="Normalized output directory.")
    parser.add_argument("--tools-config", type=Path, default=Path("configs/tools.yaml"), help="Tool config YAML path.")
    parser.add_argument("--out", type=Path, default=Path("outputs/flowchart_graph/review/index.html"), help="Review HTML path.")
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("outputs/flowchart_graph/reports/flowchart_manual_scores.csv"),
        help="Manual scoring CSV path.",
    )
    return parser


def load_tool_order(config_path: Path) -> list[str]:
    config = load_yaml(config_path)
    return list(config.keys())


def merge_score_rows(score_path: Path, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    existing_rows = {(row.get("crop_id", ""), row.get("tool", "")): row for row in read_csv_rows(score_path)}
    merged: list[dict[str, str]] = []
    for crop_id, tool in pairs:
        row = existing_rows.get((crop_id, tool), {})
        merged.append(
            {
                "crop_id": crop_id,
                "tool": tool,
                "node_score": row.get("node_score", ""),
                "edge_score": row.get("edge_score", ""),
                "direction_score": row.get("direction_score", ""),
                "label_score": row.get("label_score", ""),
                "mermaid_valid": row.get("mermaid_valid", ""),
                "overall_score": row.get("overall_score", ""),
                "major_errors": row.get("major_errors", ""),
            }
        )
    return merged


def status_badge(status: str) -> str:
    color = {
        "success": "#0f766e",
        "partial": "#b45309",
        "skipped": "#475569",
        "failed": "#b91c1c",
    }.get(status, "#475569")
    return f'<span class="status" style="background:{color}">{html.escape(status)}</span>'


def render_tool_panel(tool: str, payload: dict[str, object]) -> str:
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    mermaid = str(payload.get("mermaid", "") or "")
    raw_output = str(payload.get("raw_output", "") or "")
    parse_errors = payload.get("parse_errors", [])
    status = str(payload.get("status", "") or "")
    reason = str(payload.get("run_reason", "") or "")
    render_success = bool(mermaid.strip()) and not parse_errors
    mermaid_block = (
        f'<div class="mermaid">{html.escape(mermaid)}</div>'
        if render_success
        else '<div class="mermaid-fallback">无法渲染 Mermaid</div>'
    )
    return f"""
    <section class="tool-card">
      <h3>{html.escape(tool)} {status_badge(status)}</h3>
      <p class="reason">{html.escape(reason)}</p>
      <p class="render-status">mermaid render: {"success" if render_success else "failed"}</p>
      <div class="tool-grid">
        <div>
          <h4>nodes</h4>
          <pre>{html.escape(json.dumps(nodes, ensure_ascii=False, indent=2))}</pre>
        </div>
        <div>
          <h4>edges</h4>
          <pre>{html.escape(json.dumps(edges, ensure_ascii=False, indent=2))}</pre>
        </div>
      </div>
      <h4>mermaid</h4>
      <pre>{html.escape(mermaid)}</pre>
      {mermaid_block}
      <h4>raw output</h4>
      <pre>{html.escape(raw_output)}</pre>
      <h4>parse errors</h4>
      <pre>{html.escape(json.dumps(parse_errors, ensure_ascii=False, indent=2))}</pre>
    </section>
    """


def main() -> None:
    args = build_parser().parse_args()
    ensure_flowchart_dirs()
    crops_dir = (PROJECT_ROOT / args.crops_dir).resolve()
    normalized_dir = (PROJECT_ROOT / args.normalized_dir).resolve()
    config_path = (PROJECT_ROOT / args.tools_config).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()
    score_path = (PROJECT_ROOT / args.scores).resolve()

    crop_records = build_crop_records(crops_dir)
    if not crop_records:
        console.print("[yellow]请先放入流程图 crop 图片。[/yellow]")
        return

    tool_order = load_tool_order(config_path)
    pair_keys: list[tuple[str, str]] = []
    sections: list[str] = []

    for crop in crop_records:
        crop_id = crop["crop_id"]
        image_path = (PROJECT_ROOT / crop["image_path"]).resolve()
        panels: list[str] = []
        for tool in tool_order:
            pair_keys.append((crop_id, tool))
            normalized_path = normalized_dir / tool / f"{crop_id}.json"
            if normalized_path.exists():
                payload = read_json(normalized_path)
            else:
                payload = {
                    "crop_id": crop_id,
                    "tool": tool,
                    "nodes": [],
                    "edges": [],
                    "mermaid": "",
                    "raw_output": "",
                    "status": "skipped",
                    "parse_errors": ["缺少标准化输出文件。"],
                    "run_reason": "missing normalized output",
                }
            panels.append(render_tool_panel(tool, payload))

        sections.append(
            f"""
            <section class="crop-section" id="{html.escape(crop_id)}">
              <div class="crop-header">
                <h2>{html.escape(crop_id)}</h2>
                <p>{html.escape(crop['relative_image_path'])}</p>
              </div>
              <div class="crop-layout">
                <div class="crop-preview">
                  <img src="{image_data_uri(image_path)}" alt="{html.escape(crop_id)}" />
                </div>
                <div class="tool-panels">
                  {''.join(panels)}
                </div>
              </div>
            </section>
            """
        )

    ensure_parent(out_path)
    out_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Flowchart Graph Benchmark Review</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --panel: #fffdf8;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #d6d3d1;
      --accent: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Noto Sans SC", "PingFang SC", sans-serif; background: linear-gradient(180deg, #f6f4ee 0%, #efe8dc 100%); color: var(--text); }}
    header {{ padding: 24px 32px 12px; }}
    main {{ padding: 0 24px 32px; }}
    .crop-section {{ background: rgba(255,255,255,0.72); backdrop-filter: blur(4px); border: 1px solid var(--border); border-radius: 20px; margin: 20px 0; overflow: hidden; box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08); }}
    .crop-header {{ padding: 20px 24px 0; }}
    .crop-layout {{ display: grid; grid-template-columns: minmax(320px, 420px) 1fr; gap: 20px; padding: 20px 24px 24px; align-items: start; }}
    .crop-preview {{ position: sticky; top: 20px; background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; }}
    .crop-preview img {{ width: 100%; height: auto; border-radius: 12px; display: block; background: white; }}
    .tool-panels {{ display: grid; gap: 16px; }}
    .tool-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; }}
    .tool-card h3, .tool-card h4 {{ margin: 0 0 8px; }}
    .tool-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #111827; color: #f9fafb; border-radius: 12px; padding: 12px; overflow: auto; font-size: 12px; line-height: 1.45; }}
    .status {{ display: inline-block; color: white; padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
    .reason, .render-status {{ color: var(--muted); margin: 4px 0 12px; }}
    .mermaid, .mermaid-fallback {{ background: white; border: 1px dashed var(--border); border-radius: 12px; padding: 12px; margin-bottom: 12px; min-height: 84px; }}
    @media (max-width: 1080px) {{
      .crop-layout {{ grid-template-columns: 1fr; }}
      .crop-preview {{ position: static; }}
      .tool-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Flowchart Graph Benchmark Review</h1>
    <p>左侧是输入 crop，右侧按工具展示 nodes、edges、Mermaid、raw output 和状态。</p>
  </header>
  <main>
    {''.join(sections)}
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, securityLevel: "loose" }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )

    score_rows = merge_score_rows(score_path, pair_keys)
    write_csv(score_path, score_rows, SCORE_FIELDS)

    console.print(f"[green]已生成 review HTML：[/green]{relative_path(out_path, PROJECT_ROOT)}")
    console.print(f"[green]已生成人工评分 CSV：[/green]{relative_path(score_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
