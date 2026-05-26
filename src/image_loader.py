from __future__ import annotations

import hashlib
from pathlib import Path

from src.schema import ImageTask

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _build_image_id(relative_path: Path) -> str:
    digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()
    return digest[:16]


def load_image_tasks(data_dir: Path) -> list[ImageTask]:
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    tasks: list[ImageTask] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = path.relative_to(data_dir)
        tasks.append(
            ImageTask(
                image_id=_build_image_id(relative_path),
                image_path=str(path),
                file_name=path.name,
                file_ext=path.suffix.lower(),
            )
        )

    return tasks

