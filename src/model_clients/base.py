from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from src.schema import ImageTask, ModelOutput


class BaseVLMClient(ABC):
    def __init__(self, model_name: str, config: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self.config = config or {}

    def generate(self, image_task: ImageTask, prompt: str) -> ModelOutput:
        start = perf_counter()
        try:
            result = self._generate_impl(image_task=image_task, prompt=prompt)
            if isinstance(result, ModelOutput):
                output = result
            else:
                output = ModelOutput(
                    image_id=image_task.image_id,
                    model_name=self.model_name,
                    success=bool(result.get("success", False)),
                    raw_text=str(result.get("raw_text", "")),
                    parsed=result.get("parsed"),
                    error=result.get("error"),
                    latency_ms=result.get("latency_ms"),
                    vendor=self._normalize_optional_text(result.get("vendor")),
                    source_type=self._normalize_optional_text(result.get("source_type")),
                )
        except Exception as exc:
            output = ModelOutput(
                image_id=image_task.image_id,
                model_name=self.model_name,
                success=False,
                raw_text="",
                error=f"{type(exc).__name__}: {exc}",
            )

        output.image_id = image_task.image_id
        output.model_name = self.model_name
        if not output.vendor:
            output.vendor = self._default_vendor()
        if not output.source_type:
            output.source_type = self._default_source_type()
        if output.latency_ms is None:
            output.latency_ms = int((perf_counter() - start) * 1000)
        return output

    @abstractmethod
    def _generate_impl(
        self, image_task: ImageTask, prompt: str
    ) -> ModelOutput | dict[str, Any]:
        raise NotImplementedError

    def _default_vendor(self) -> str | None:
        return self._normalize_optional_text(
            self.config.get("vendor") or self.config.get("provider")
        )

    def _default_source_type(self) -> str | None:
        configured = self._normalize_optional_text(self.config.get("source_type"))
        if configured:
            return configured
        provider = self._normalize_optional_text(self.config.get("provider"))
        if provider == "mock":
            return "mock"
        return None

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None
