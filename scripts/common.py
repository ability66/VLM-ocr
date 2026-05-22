from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from rich.console import Console


console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def relative_path(path: Path, start: Path | None = None) -> str:
    base = start or PROJECT_ROOT
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def filesystem_relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def slugify(value: str, max_length: int = 40) -> str:
    original = value.strip()
    value = re.sub(r"[^0-9A-Za-z._-]+", "-", original).strip("-._")
    if not value:
        digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
        return f"item-{digest}"
    return value[:max_length]


def stable_sample_id(pdf_path: Path, page_index: int, pdf_root: Path) -> str:
    rel = relative_path(pdf_path, pdf_root)
    digest = hashlib.sha1(f"{rel}|{page_index}".encode("utf-8")).hexdigest()[:10]
    stem = slugify(pdf_path.stem, max_length=28)
    return f"{stem}-p{page_index:04d}-{digest}"


def list_pdf_files(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(
        path
        for path in pdf_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(normalized)


def write_empty_csv(path: Path, fieldnames: Sequence[str]) -> None:
    write_csv(path, [], fieldnames)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层结构必须是 dict: {path}")
    return data


def list_image_files(root: Path, recursive: bool = True, exclude_dir_names: Sequence[str] | None = None) -> list[Path]:
    if not root.exists():
        return []
    exclude = set(exclude_dir_names or [])
    extensions = {".png", ".jpg", ".jpeg"}
    paths = root.rglob("*") if recursive else root.glob("*")
    results: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if any(part in exclude for part in path.relative_to(root).parts[:-1]):
            continue
        results.append(path)
    return sorted(results)


def markdown_table(rows: Sequence[dict[str, Any]], headers: Sequence[str]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines: list[str] = []
    for row in rows:
        cells = [str(row.get(header, "")).replace("\n", "<br>") for header in headers]
        body_lines.append("| " + " | ".join(cells) + " |")
    if not body_lines:
        body_lines.append("| " + " | ".join([""] * len(headers)) + " |")
    return "\n".join([header_line, separator, *body_lines])
