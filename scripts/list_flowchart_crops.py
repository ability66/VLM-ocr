from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, console, relative_path, write_csv, write_empty_csv
from flowchart_graph_common import FLOWCHART_CROPS_DIR, FLOWCHART_REPORTS_DIR, build_crop_records, ensure_flowchart_dirs


FIELDS = ["crop_id", "image_path", "source_pdf", "page_number", "notes"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan flowchart crop images and build an input manifest CSV.")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/flowchart_crops"), help="Directory containing flowchart crop images.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/flowchart_graph/reports/flowchart_inputs.csv"),
        help="Output CSV path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_flowchart_dirs()
    crops_dir = (PROJECT_ROOT / args.crops_dir).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()

    records = build_crop_records(crops_dir)
    if not records:
        write_empty_csv(out_path, FIELDS)
        console.print("[yellow]请先放入流程图 crop 图片。[/yellow]")
        console.print(f"[yellow]已生成空输入清单：{relative_path(out_path, PROJECT_ROOT)}[/yellow]")
        return

    rows = [
        {
            "crop_id": record["crop_id"],
            "image_path": record["image_path"],
            "source_pdf": "",
            "page_number": "",
            "notes": "",
        }
        for record in records
    ]
    write_csv(out_path, rows, FIELDS)
    console.print(f"[green]已生成流程图输入清单：[/green]{relative_path(out_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
