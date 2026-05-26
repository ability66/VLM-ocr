from __future__ import annotations

import base64
import os
from importlib import import_module
from pathlib import Path
from typing import Any

from src.model_clients.base import BaseVLMClient
from src.schema import ImageTask

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class OpenAICompatibleVLMClient(BaseVLMClient):
    def __init__(self, model_name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, config=config)
        self.name = self._read_text_config("name", fallback=model_name)
        self.base_url = self._read_text_config("base_url")
        self.api_key_env = self._read_text_config("api_key_env")
        self.vendor = self._read_text_config("vendor")
        self.source_type = self._read_text_config("source_type")
        self.supports_vision = self._read_bool_config("supports_vision", default=True)
        self.timeout = self._read_int_config("timeout", default=120)
        self.max_tokens = self._read_int_config("max_tokens", default=4096)
        self.temperature = self._read_float_config("temperature", default=0.0)

    def _generate_impl(self, image_task: ImageTask, prompt: str) -> dict[str, Any]:
        try:
            return self._generate_impl_with_error_capture(image_task=image_task, prompt=prompt)
        except Exception as exc:
            return {
                "success": False,
                "raw_text": "",
                "error": f"{type(exc).__name__}: {exc}",
                "vendor": self.vendor,
                "source_type": self.source_type,
            }

    def _generate_impl_with_error_capture(
        self, image_task: ImageTask, prompt: str
    ) -> dict[str, Any]:
        if not self.supports_vision:
            return {
                "success": False,
                "raw_text": "",
                "error": "Model config does not support vision input",
                "vendor": self.vendor,
                "source_type": self.source_type,
            }

        if not self.base_url:
            return {
                "success": False,
                "raw_text": "",
                "error": "Missing base_url in model config",
                "vendor": self.vendor,
                "source_type": self.source_type,
            }

        if not self.api_key_env:
            return {
                "success": False,
                "raw_text": "",
                "error": "Missing api_key_env in model config",
                "vendor": self.vendor,
                "source_type": self.source_type,
            }

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return {
                "success": False,
                "raw_text": "",
                "error": f"Missing API key: set {self.api_key_env}",
                "vendor": self.vendor,
                "source_type": self.source_type,
            }

        try:
            openai_module = import_module("openai")
        except ImportError:
            return {
                "success": False,
                "raw_text": "",
                "error": 'OpenAI SDK is not installed. Install it with: uv pip install -e ".[api]"',
                "vendor": self.vendor,
                "source_type": self.source_type,
            }

        client = openai_module.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        data_url = self._build_image_data_url(Path(image_task.image_path))
        response = client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        raw_content = response.choices[0].message.content
        raw_text = raw_content if isinstance(raw_content, str) else str(raw_content)
        return {
            "success": True,
            "raw_text": raw_text,
            "error": None,
            "vendor": self.vendor,
            "source_type": self.source_type,
        }

    def _build_image_data_url(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime_type = _MIME_TYPES.get(suffix)
        if mime_type is None:
            raise ValueError(f"Unsupported image suffix for OpenAI-compatible client: {suffix}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _read_text_config(self, key: str, fallback: str | None = None) -> str | None:
        value = self.config.get(key, fallback)
        if value is None:
            return None
        text = str(value).strip()
        return text or fallback

    def _read_bool_config(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _read_int_config(self, key: str, default: int) -> int:
        value = self.config.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _read_float_config(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
