from __future__ import annotations

import argparse
from pathlib import Path

import fitz

from common import (
    DATA_DIR,
    PROJECT_ROOT,
    console,
    ensure_dir,
    list_pdf_files,
    relative_path,
    slugify,
    write_csv,
    write_empty_csv,
)


MANIFEST_FIELDS = [
    "render_id",
    "pdf_path",
    "pdf_name",
    "pdf_stem",
    "page_index",
    "page_number",
    "image_path",
    "dpi",
    "width_px",
    "height_px",
    "status",
    "error_message",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render PDF pages into PNG images for downstream flowchart cropping.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"), help="Directory containing PDF files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/page_images"),
        help="Directory for rendered PNG pages.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/page_images/page_images_manifest.csv"),
        help="Manifest CSV path.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Rendering DPI.")
    parser.add_argument(
        "--page-limit",
        type=int,
        default=0,
        help="Optional maximum number of pages to render per PDF. 0 means all pages.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pdf_dir = (PROJECT_ROOT / args.pdf_dir).resolve()
    out_dir = ensure_dir((PROJECT_ROOT / args.out_dir).resolve())
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()

    ensure_dir(DATA_DIR / "page_images")
    pdf_paths = list_pdf_files(pdf_dir)
    if not pdf_paths:
        write_empty_csv(manifest_path, MANIFEST_FIELDS)
        console.print(f"[yellow]未在 {relative_path(pdf_dir, PROJECT_ROOT)} 找到 PDF。已生成空 manifest：{relative_path(manifest_path, PROJECT_ROOT)}[/yellow]")
        return

    scale = args.dpi / 72.0
    manifest_rows: list[dict[str, str]] = []
    for pdf_path in pdf_paths:
        pdf_stem = slugify(pdf_path.stem, max_length=80)
        pdf_out_dir = ensure_dir(out_dir / pdf_stem)
        try:
            with fitz.open(pdf_path) as document:
                page_total = len(document)
                limit = page_total if args.page_limit <= 0 else min(page_total, args.page_limit)
                for page_index in range(limit):
                    page_number = page_index + 1
                    render_id = f"{pdf_stem}-p{page_index:04d}"
                    image_path = pdf_out_dir / f"{render_id}.png"
                    manifest_row = {
                        "render_id": render_id,
                        "pdf_path": relative_path(pdf_path, PROJECT_ROOT),
                        "pdf_name": pdf_path.name,
                        "pdf_stem": pdf_stem,
                        "page_index": str(page_index),
                        "page_number": str(page_number),
                        "image_path": relative_path(image_path, PROJECT_ROOT),
                        "dpi": str(args.dpi),
                        "width_px": "",
                        "height_px": "",
                        "status": "success",
                        "error_message": "",
                    }
                    try:
                        page = document.load_page(page_index)
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                        pixmap.save(image_path)
                        manifest_row["width_px"] = str(pixmap.width)
                        manifest_row["height_px"] = str(pixmap.height)
                    except Exception as exc:
                        manifest_row["status"] = "failed"
                        manifest_row["error_message"] = str(exc)
                    manifest_rows.append(manifest_row)
        except Exception as exc:
            manifest_rows.append(
                {
                    "render_id": f"{pdf_stem}-p0000",
                    "pdf_path": relative_path(pdf_path, PROJECT_ROOT),
                    "pdf_name": pdf_path.name,
                    "pdf_stem": pdf_stem,
                    "page_index": "",
                    "page_number": "",
                    "image_path": "",
                    "dpi": str(args.dpi),
                    "width_px": "",
                    "height_px": "",
                    "status": "failed",
                    "error_message": str(exc),
                }
            )

    write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
    console.print(f"[green]已写入页面图片 manifest：[/green]{relative_path(manifest_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
