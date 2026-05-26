from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from src.model_clients.base import BaseVLMClient
from src.schema import ImageTask


class DashScopeClient(BaseVLMClient):
    def _generate_impl(self, image_task: ImageTask, prompt: str) -> dict[str, Any]:
        api_key = self._resolve_api_key()
        if not api_key:
            return {
                "success": False,
                "raw_text": "",
                "error": "Missing API key: set DASHSCOPE_API_KEY or QWEN_API_KEY",
            }

        image_b64 = self._encode_image(Path(image_task.image_path))
        _request_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "image_base64": image_b64,
        }

        try:
            import dashscope  # type: ignore  # noqa: F401
        except ImportError:
            return {
                "success": False,
                "raw_text": "",
                "error": "DashScope SDK is not installed. DashScope real API call is not fully implemented yet",
            }

        return {
            "success": False,
            "raw_text": "",
            "error": "DashScope real API call is not fully implemented yet",
        }

    def _resolve_api_key(self) -> str | None:
        preferred = os.getenv("DASHSCOPE_API_KEY")
        fallback = os.getenv("QWEN_API_KEY")
        return preferred or fallback

    def _encode_image(self, image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

