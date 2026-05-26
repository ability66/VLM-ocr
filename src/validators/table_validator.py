from __future__ import annotations


def parse_markdown_table(content: str) -> dict:
    stripped = content.strip()
    if not stripped:
        return {
            "looks_like_table": False,
            "row_count": 0,
            "col_count": 0,
            "rows": [],
            "cell_texts": [],
            "empty_cell_ratio": 0.0,
            "stable_columns": False,
            "separator_present": False,
        }

    rows: list[list[str]] = []
    separator_present = False
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_separator_row(line):
            separator_present = True
            continue
        cells = _split_table_row(line)
        if not cells:
            continue
        rows.append(cells)

    row_count = len(rows)
    col_lengths = [len(row) for row in rows if row]
    col_count = max(col_lengths, default=0)
    stable_columns = bool(col_lengths) and len(set(col_lengths)) == 1

    flattened_cells = [cell for row in rows for cell in row]
    empty_cells = sum(1 for cell in flattened_cells if not cell.strip())
    empty_cell_ratio = empty_cells / len(flattened_cells) if flattened_cells else 0.0

    pipe_line_count = sum(1 for line in stripped.splitlines() if "|" in line)
    csv_like = _looks_like_csv(rows=rows, raw_content=stripped)
    looks_like_table = (
        row_count > 0
        and col_count > 1
        and (separator_present or pipe_line_count >= 2 or csv_like)
    )

    return {
        "looks_like_table": looks_like_table,
        "row_count": row_count,
        "col_count": col_count,
        "rows": rows,
        "cell_texts": [cell.strip() for cell in flattened_cells if cell.strip()],
        "empty_cell_ratio": round(empty_cell_ratio, 4),
        "stable_columns": stable_columns,
        "separator_present": separator_present,
    }


def validate_table(content: str, image_type: str | None = None) -> dict:
    stripped = content.strip()
    errors: list[str] = []
    warnings: list[str] = []

    if not stripped:
        if image_type in {"chart", "table"}:
            errors.append("image_type is chart/table but table content is empty")
        errors.append("kind is table but content is empty")
        return {
            "valid_basic": False,
            "row_count": 0,
            "col_count": 0,
            "cell_texts": [],
            "empty_cell_ratio": 0.0,
            "errors": _deduplicate(errors),
            "warnings": warnings,
            "score": 0.0,
        }

    parsed = parse_markdown_table(stripped)
    if not parsed["looks_like_table"]:
        errors.append("kind is table but content is not a recognizable markdown/csv table")
    if parsed["row_count"] < 1 or parsed["col_count"] < 1:
        errors.append("table content has no valid rows or columns")
    if not parsed["stable_columns"] and parsed["row_count"] > 1:
        warnings.append("table rows have inconsistent column counts")
    if parsed["row_count"] == 1:
        warnings.append("table content has only one parsed row")
    if parsed["empty_cell_ratio"] >= 0.5:
        warnings.append("table content has a high empty cell ratio")

    score = 1.0
    if not parsed["looks_like_table"]:
        score -= 0.45
    if parsed["row_count"] < 1 or parsed["col_count"] < 1:
        score -= 0.4
    if not parsed["stable_columns"] and parsed["row_count"] > 1:
        score -= 0.15
    if parsed["row_count"] == 1:
        score -= 0.1
    if parsed["empty_cell_ratio"] >= 0.5:
        score -= 0.15

    return {
        "valid_basic": bool(stripped) and not errors,
        "row_count": parsed["row_count"],
        "col_count": parsed["col_count"],
        "cell_texts": parsed["cell_texts"],
        "empty_cell_ratio": parsed["empty_cell_ratio"],
        "errors": _deduplicate(errors),
        "warnings": _deduplicate(warnings),
        "score": round(max(0.0, min(1.0, score)), 4),
    }


def _split_table_row(line: str) -> list[str]:
    if "|" in line:
        return [cell.strip() for cell in line.strip("|").split("|")]
    if "," in line:
        return [cell.strip() for cell in line.split(",")]
    return [line.strip()]


def _is_separator_row(line: str) -> bool:
    stripped = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return not stripped and ("-" in line or "|" in line)


def _looks_like_csv(rows: list[list[str]], raw_content: str) -> bool:
    if "," not in raw_content:
        return False
    if len(rows) < 2:
        return False
    return all(len(row) > 1 for row in rows)


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
