from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from common import (
    DATA_DIR,
    OUTPUTS_DIR,
    ensure_dir,
    list_image_files,
    read_text,
    relative_path,
    slugify,
    write_json,
)


FLOWCHART_CROPS_DIR = DATA_DIR / "flowchart_crops"
FLOWCHART_OUTPUTS_DIR = OUTPUTS_DIR / "flowchart_graph"
FLOWCHART_RAW_DIR = FLOWCHART_OUTPUTS_DIR / "raw"
FLOWCHART_NORMALIZED_DIR = FLOWCHART_OUTPUTS_DIR / "normalized"
FLOWCHART_REVIEW_DIR = FLOWCHART_OUTPUTS_DIR / "review"
FLOWCHART_REPORTS_DIR = FLOWCHART_OUTPUTS_DIR / "reports"
FLOWCHART_MANUAL_MERMAID_DIR = FLOWCHART_CROPS_DIR / "manual_mermaid"


def ensure_flowchart_dirs() -> None:
    ensure_dir(FLOWCHART_CROPS_DIR)
    ensure_dir(FLOWCHART_MANUAL_MERMAID_DIR)
    ensure_dir(FLOWCHART_RAW_DIR)
    ensure_dir(FLOWCHART_NORMALIZED_DIR)
    ensure_dir(FLOWCHART_REVIEW_DIR)
    ensure_dir(FLOWCHART_REPORTS_DIR)


def build_crop_records(crops_dir: Path = FLOWCHART_CROPS_DIR) -> list[dict[str, str]]:
    image_paths = list_image_files(crops_dir, recursive=True, exclude_dir_names=["manual_mermaid"])
    stem_counts: dict[str, int] = {}
    for image_path in image_paths:
        stem_counts[image_path.stem] = stem_counts.get(image_path.stem, 0) + 1

    records: list[dict[str, str]] = []
    for image_path in image_paths:
        rel = image_path.resolve().relative_to(crops_dir.resolve())
        crop_id = image_path.stem
        if stem_counts[image_path.stem] > 1 or len(rel.parts) > 1:
            digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:8]
            crop_id = f"{slugify(image_path.stem, max_length=48)}-{digest}"
        records.append(
            {
                "crop_id": crop_id,
                "image_path": relative_path(image_path),
                "image_name": image_path.name,
                "relative_image_path": rel.as_posix(),
            }
        )
    return records


def write_run_status(
    out_dir: Path,
    *,
    tool: str,
    crop_id: str,
    status: str,
    reason: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool": tool,
        "crop_id": crop_id,
        "status": status,
        "reason": reason,
        "details": details or {},
    }
    write_json(out_dir / "run_status.json", payload)
    return payload


def load_run_status(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "run_status.json"
    if not path.exists():
        return {"status": "failed", "reason": "missing run_status.json", "details": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def image_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", stripped, count=1)
        stripped = re.sub(r"\n?```$", "", stripped, count=1)
    return stripped.strip()


def extract_json_candidate(text: str) -> str:
    stripped = strip_code_fence(text)
    fenced = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def load_json_candidate(text: str) -> Any:
    candidate = extract_json_candidate(text)
    return json.loads(candidate)


def node_shape_from_wrapper(wrapper: str) -> str:
    mapping = {
        "circle": "circle",
        "stadium": "stadium",
        "round": "round",
        "decision": "decision",
        "parallelogram": "parallelogram",
        "hexagon": "hexagon",
        "rect": "rect",
    }
    return mapping.get(wrapper, "rect")


def parse_mermaid_node(expr: str) -> tuple[dict[str, Any] | None, str | None]:
    token = expr.strip().rstrip(";").strip()
    if not token:
        return None, "empty node token"
    match = re.match(r"^([A-Za-z0-9_:.\\/-]+)\s*(.*)$", token)
    if not match:
        return None, f"unrecognized node token: {token}"
    node_id = match.group(1)
    remainder = match.group(2).strip()
    node: dict[str, Any] = {"id": node_id, "text": node_id, "shape": "rect", "bbox": None}
    if not remainder:
        return node, None

    wrappers = [
        ("((", "))", "circle"),
        ("([", "])", "stadium"),
        ("{", "}", "decision"),
        ("[/", "/]", "parallelogram"),
        ("[\\", "\\]", "parallelogram"),
        ("{{", "}}", "hexagon"),
        ("(", ")", "round"),
        ("[", "]", "rect"),
    ]
    for prefix, suffix, shape in wrappers:
        if remainder.startswith(prefix) and remainder.endswith(suffix):
            label = remainder[len(prefix) : len(remainder) - len(suffix)].strip().strip('"').strip("'")
            if label:
                node["text"] = label
            node["shape"] = node_shape_from_wrapper(shape)
            return node, None

    clean = remainder.strip('"').strip("'")
    if clean:
        node["text"] = clean
    return node, None


def connector_to_direction(connector: str) -> tuple[str, bool]:
    if "<" in connector and ">" in connector:
        return "bidirectional", False
    if connector.startswith("<") and ">" not in connector:
        return "forward", True
    if ">" in connector:
        return "forward", False
    return "undirected", False


def parse_mermaid_flowchart(mermaid_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    text = strip_code_fence(mermaid_text)
    text = text.replace("\r\n", "\n")
    lines: list[str] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        if raw_line.startswith("%%"):
            continue
        pieces = [piece.strip() for piece in raw_line.split(";") if piece.strip()]
        lines.extend(pieces or [raw_line])

    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    errors: list[str] = []
    edge_re = re.compile(
        r"^(?P<left>.+?)\s*(?P<connector><<?[-.=ox]+>?)(?:\|(?P<label>[^|]+)\|)?\s*(?P<right>.+)$"
    )

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("flowchart ") or lowered.startswith("graph "):
            continue
        if lowered.startswith("subgraph ") or lowered == "end":
            continue
        if lowered.startswith("classdef ") or lowered.startswith("class ") or lowered.startswith("style "):
            continue
        if lowered.startswith("linkstyle "):
            continue

        edge_match = edge_re.match(line)
        if edge_match:
            left_node, left_error = parse_mermaid_node(edge_match.group("left"))
            right_node, right_error = parse_mermaid_node(edge_match.group("right"))
            if left_node is None or right_node is None:
                errors.append(left_error or right_error or f"无法解析边: {line}")
                continue
            nodes_by_id[left_node["id"]] = {**nodes_by_id.get(left_node["id"], {}), **left_node}
            nodes_by_id[right_node["id"]] = {**nodes_by_id.get(right_node["id"], {}), **right_node}

            connector = edge_match.group("connector")
            direction, reverse = connector_to_direction(connector)
            source = right_node["id"] if reverse else left_node["id"]
            target = left_node["id"] if reverse else right_node["id"]
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "label": (edge_match.group("label") or "").strip(),
                    "direction": direction,
                    "connector": connector,
                }
            )
            continue

        node, error = parse_mermaid_node(line)
        if node is not None:
            nodes_by_id[node["id"]] = {**nodes_by_id.get(node["id"], {}), **node}
        else:
            errors.append(error or f"无法解析 Mermaid 行: {line}")

    return list(nodes_by_id.values()), edges, errors


def mermaid_node_token(node: dict[str, Any]) -> str:
    node_id = str(node.get("id", "")).strip() or "node"
    text = str(node.get("text", node_id)).strip().replace('"', "'")
    shape = str(node.get("shape", "rect") or "rect").lower()
    if shape in {"circle"}:
        return f'{node_id}(("{text}"))'
    if shape in {"stadium", "pill"}:
        return f'{node_id}(["{text}"])'
    if shape in {"round", "rounded"}:
        return f'{node_id}("{text}")'
    if shape in {"decision", "diamond"}:
        return f'{node_id}{{"{text}"}}'
    if shape in {"parallelogram", "input", "output"}:
        return f'{node_id}[/"{text}"/]'
    if shape in {"hexagon"}:
        return f'{node_id}{{{{"{text}"}}}}'
    return f'{node_id}["{text}"]'


def mermaid_connector(direction: str, label: str = "") -> str:
    direction_value = (direction or "forward").lower()
    connector = "---"
    if direction_value == "forward":
        connector = "-->"
    elif direction_value == "bidirectional":
        connector = "<-->"
    elif direction_value == "undirected":
        connector = "---"
    if label:
        connector = f"{connector}|{label}|"
    return connector


def mermaid_from_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    seen: set[str] = set()
    lines = ["flowchart TD"]
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        lines.append(f"    {mermaid_node_token(node)}")
    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        if source not in seen:
            seen.add(source)
            lines.append(f'    {source}["{source}"]')
        if target not in seen:
            seen.add(target)
            lines.append(f'    {target}["{target}"]')
        connector = mermaid_connector(str(edge.get("direction", "forward")), str(edge.get("label", "")).strip())
        lines.append(f"    {source} {connector} {target}")
    return "\n".join(lines)


def normalize_graph_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    mermaid = str(payload.get("mermaid") or "").strip()
    if not mermaid and (nodes or edges):
        mermaid = mermaid_from_graph(nodes, edges)
    return nodes, edges, mermaid


def pick_raw_text(raw_dir: Path) -> str:
    preferred_patterns = [
        "raw_output.mmd",
        "raw_output.json",
        "raw_output.md",
        "raw_output.txt",
        "content_list.json",
        "stdout.txt",
        "stderr.txt",
    ]
    for name in preferred_patterns:
        path = raw_dir / name
        if path.exists():
            return read_text(path)
    candidates = sorted(path for path in raw_dir.iterdir() if path.is_file() and path.name != "run_status.json")
    for path in candidates:
        if path.suffix.lower() in {".mmd", ".json", ".md", ".txt"}:
            return read_text(path)
    return ""

