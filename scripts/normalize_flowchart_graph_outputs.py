from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, console, load_yaml, read_text, relative_path, write_json
from flowchart_graph_common import (
    FLOWCHART_CROPS_DIR,
    FLOWCHART_NORMALIZED_DIR,
    FLOWCHART_RAW_DIR,
    build_crop_records,
    ensure_flowchart_dirs,
    extract_json_candidate,
    load_run_status,
    mermaid_from_graph,
    normalize_graph_payload,
    parse_mermaid_flowchart,
    pick_raw_text,
    strip_code_fence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize raw flowchart graph outputs into a shared schema.")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/flowchart_crops"), help="Directory containing flowchart crop images.")
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/flowchart_graph/raw"), help="Raw output directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowchart_graph/normalized"), help="Normalized output directory.")
    parser.add_argument("--tools-config", type=Path, default=Path("configs/tools.yaml"), help="Tool config YAML path.")
    parser.add_argument("--tools", type=str, default="", help="Optional comma-separated tool list.")
    return parser


def selected_tools(config_path: Path, requested: str) -> list[str]:
    config = load_yaml(config_path)
    tools = list(config.keys())
    if requested.strip():
        requested_tools = [item.strip() for item in requested.split(",") if item.strip()]
        return [tool for tool in requested_tools if tool in config]
    return tools


def looks_like_mermaid(text: str) -> bool:
    stripped = strip_code_fence(text)
    lowered = stripped.lower()
    return lowered.startswith("flowchart") or lowered.startswith("graph") or "-->" in stripped or "---" in stripped


def normalize_from_json(data: Any, parse_errors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if isinstance(data, dict):
        if isinstance(data.get("result"), dict):
            data = data["result"]
        elif isinstance(data.get("graph"), dict):
            data = data["graph"]

    if isinstance(data, dict):
        nodes, edges, mermaid = normalize_graph_payload(data)
        if not mermaid and isinstance(data.get("mermaid_code"), str):
            mermaid = data["mermaid_code"].strip()
        if not nodes and not edges and mermaid:
            nodes, edges, mermaid_errors = parse_mermaid_flowchart(mermaid)
            parse_errors.extend(mermaid_errors)
        return nodes, edges, mermaid

    parse_errors.append("JSON 顶层结构不是 object，无法直接归一化。")
    return [], [], ""


def normalize_one(raw_dir: Path, crop_id: str, tool: str) -> dict[str, Any]:
    run_status = load_run_status(raw_dir) if raw_dir.exists() else {"status": "skipped", "reason": "missing raw directory", "details": {}}
    status = str(run_status.get("status", "failed"))
    raw_output = pick_raw_text(raw_dir) if raw_dir.exists() else ""
    parse_errors: list[str] = []
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    mermaid = ""

    if status not in {"skipped", "failed"} and raw_output.strip():
        json_path = raw_dir / "raw_output.json"
        raw_json: Any | None = None
        if json_path.exists():
            try:
                raw_json = json.loads(read_text(json_path))
            except Exception as exc:
                parse_errors.append(f"raw_output.json 解析失败: {exc}")
        else:
            try:
                raw_json = json.loads(extract_json_candidate(raw_output))
            except Exception:
                raw_json = None

        if raw_json is not None:
            nodes, edges, mermaid = normalize_from_json(raw_json, parse_errors)

        if not nodes and not edges and not mermaid and looks_like_mermaid(raw_output):
            mermaid = strip_code_fence(raw_output)
            parsed_nodes, parsed_edges, mermaid_errors = parse_mermaid_flowchart(mermaid)
            nodes = parsed_nodes
            edges = parsed_edges
            parse_errors.extend(mermaid_errors)

        if not mermaid and (nodes or edges):
            mermaid = mermaid_from_graph(nodes, edges)

    normalized_status = status
    if status == "success" and not (nodes or edges or mermaid):
        normalized_status = "partial"
        if not parse_errors:
            parse_errors.append("工具运行成功，但未识别出 graph JSON 或 Mermaid。")
    elif status == "success" and parse_errors:
        normalized_status = "partial"

    return {
        "crop_id": crop_id,
        "tool": tool,
        "nodes": nodes,
        "edges": edges,
        "mermaid": mermaid,
        "raw_output": raw_output,
        "status": normalized_status,
        "parse_errors": parse_errors,
        "run_reason": str(run_status.get("reason", "")),
        "run_details": run_status.get("details", {}),
    }


def main() -> None:
    args = build_parser().parse_args()
    ensure_flowchart_dirs()
    crops_dir = (PROJECT_ROOT / args.crops_dir).resolve()
    raw_dir = (PROJECT_ROOT / args.raw_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    tools = selected_tools((PROJECT_ROOT / args.tools_config).resolve(), args.tools)
    crop_records = build_crop_records(crops_dir)

    if not crop_records:
        console.print("[yellow]请先放入流程图 crop 图片。[/yellow]")
        return

    for tool in tools:
        tool_out_dir = out_dir / tool
        tool_out_dir.mkdir(parents=True, exist_ok=True)
        for crop in crop_records:
            crop_id = crop["crop_id"]
            normalized = normalize_one(raw_dir / tool / crop_id, crop_id, tool)
            write_json(tool_out_dir / f"{crop_id}.json", normalized)

    console.print(f"[green]已写入标准化结果目录：[/green]{relative_path(out_dir, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
