from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, console, markdown_table, read_json, relative_path, write_csv, write_text


CSV_FIELDS = [
    "crop_id",
    "tool",
    "status",
    "node_count",
    "edge_count",
    "mermaid_generated",
    "torch_dtype",
    "max_new_tokens",
    "reason",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize flowchart graph benchmark run results.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("outputs/flowchart_graph/raw"),
        help="Raw flowchart output directory.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/flowchart_graph/reports/flowchart_run_summary.csv"),
        help="Summary CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/flowchart_graph/reports/flowchart_run_summary.md"),
        help="Summary markdown path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = (PROJECT_ROOT / args.raw_dir).resolve()
    out_csv = (PROJECT_ROOT / args.out_csv).resolve()
    out_md = (PROJECT_ROOT / args.out_md).resolve()

    rows: list[dict[str, str]] = []
    for tool_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        tool = tool_dir.name
        for crop_dir in sorted(path for path in tool_dir.iterdir() if path.is_dir()):
            status_path = crop_dir / "run_status.json"
            if not status_path.exists():
                continue
            payload = read_json(status_path)
            details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
            rows.append(
                {
                    "crop_id": crop_dir.name,
                    "tool": tool,
                    "status": str(payload.get("status", "")),
                    "node_count": str(details.get("node_count", "")),
                    "edge_count": str(details.get("edge_count", "")),
                    "mermaid_generated": str(details.get("mermaid_generated", "")),
                    "torch_dtype": str(details.get("torch_dtype", "")),
                    "max_new_tokens": str(details.get("max_new_tokens", "")),
                    "reason": str(payload.get("reason", "")),
                }
            )

    write_csv(out_csv, rows, CSV_FIELDS)
    md = "# Flowchart Graph Run Summary\n\n"
    md += markdown_table(rows, CSV_FIELDS)
    md += "\n"
    write_text(out_md, md)

    console.print(f"[green]已生成结果汇总 CSV：[/green]{relative_path(out_csv, PROJECT_ROOT)}")
    console.print(f"[green]已生成结果汇总 Markdown：[/green]{relative_path(out_md, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
