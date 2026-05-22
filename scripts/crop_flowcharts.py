from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from common import PROJECT_ROOT, console, ensure_dir, read_csv_rows, relative_path, resolve_project_path, write_csv, write_empty_csv
from flowchart_graph_common import FLOWCHART_CROPS_DIR, FLOWCHART_REPORTS_DIR, ensure_flowchart_dirs


INPUT_FIELDS = ["crop_id", "page_image_path", "x0", "y0", "x1", "y1", "source_pdf", "page_number", "notes"]
RESULT_FIELDS = INPUT_FIELDS + ["status", "error_message", "saved_image_path"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop flowchart regions from page images using a CSV template.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/flowchart_graph/reports/flowchart_crop_template.csv"),
        help="Crop template CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/flowchart_crops"),
        help="Directory for cropped flowchart images.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/flowchart_graph/reports/flowchart_crop_results.csv"),
        help="Results CSV path.",
    )
    return parser


def to_int(value: str) -> int:
    return int(round(float(value)))


def main() -> None:
    args = build_parser().parse_args()
    ensure_flowchart_dirs()
    input_path = (PROJECT_ROOT / args.input).resolve()
    out_dir = ensure_dir((PROJECT_ROOT / args.out_dir).resolve())
    results_path = (PROJECT_ROOT / args.results).resolve()

    if not input_path.exists():
        write_empty_csv(input_path, INPUT_FIELDS)
        console.print("[yellow]未找到裁剪模板 CSV。已生成空模板，请先填写 bbox。[/yellow]")
        console.print(f"[yellow]{relative_path(input_path, PROJECT_ROOT)}[/yellow]")
        return

    rows = read_csv_rows(input_path)
    if not rows:
        write_empty_csv(results_path, RESULT_FIELDS)
        console.print("[yellow]裁剪模板为空，请先填写 bbox。[/yellow]")
        return

    result_rows: list[dict[str, str]] = []
    for row in rows:
        crop_id = row.get("crop_id", "").strip()
        page_image_path = row.get("page_image_path", "").strip()
        result = {field: row.get(field, "") for field in INPUT_FIELDS}
        result.update({"status": "success", "error_message": "", "saved_image_path": ""})

        try:
            if not crop_id:
                raise ValueError("crop_id 不能为空")
            if not page_image_path:
                raise ValueError("page_image_path 不能为空")
            image_path = resolve_project_path(page_image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"页面图片不存在: {page_image_path}")
            x0 = to_int(row.get("x0", ""))
            y0 = to_int(row.get("y0", ""))
            x1 = to_int(row.get("x1", ""))
            y1 = to_int(row.get("y1", ""))
            if x1 <= x0 or y1 <= y0:
                raise ValueError("bbox 非法，要求 x1>x0 且 y1>y0")

            with Image.open(image_path) as image:
                width, height = image.size
                left = max(0, min(width, x0))
                top = max(0, min(height, y0))
                right = max(0, min(width, x1))
                bottom = max(0, min(height, y1))
                if right <= left or bottom <= top:
                    raise ValueError("bbox 超出图片范围或裁剪后为空")
                crop = image.crop((left, top, right, bottom))
                saved_path = out_dir / f"{crop_id}.png"
                crop.save(saved_path)
                result["saved_image_path"] = relative_path(saved_path, PROJECT_ROOT)
        except Exception as exc:
            result["status"] = "failed"
            result["error_message"] = str(exc)

        result_rows.append(result)

    write_csv(results_path, result_rows, RESULT_FIELDS)
    console.print(f"[green]裁剪结果已写入：[/green]{relative_path(results_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
