from __future__ import annotations

import os
from typing import Any

from src.model_clients.base import BaseVLMClient
from src.schema import ImageTask


class _UnavailableProviderClient(BaseVLMClient):
    provider_label = "provider"
    api_key_env = ""
    sdk_import_name = ""
    sdk_error_message = ""
    not_implemented_message = ""

    def _generate_impl(self, image_task: ImageTask, prompt: str) -> dict[str, Any]:
        del image_task, prompt
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return {
                "success": False,
                "raw_text": "",
                "error": f"Missing API key: set {self.api_key_env}",
            }

        try:
            __import__(self.sdk_import_name)
        except ImportError:
            return {
                "success": False,
                "raw_text": "",
                "error": self.sdk_error_message,
            }

        return {
            "success": False,
            "raw_text": "",
            "error": self.not_implemented_message,
        }


class OpenAIClient(_UnavailableProviderClient):
    provider_label = "openai"
    api_key_env = "OPENAI_API_KEY"
    sdk_import_name = "openai"
    sdk_error_message = (
        "OpenAI SDK is not installed. OpenAI real API call is not fully implemented yet"
    )
    not_implemented_message = "OpenAI real API call is not fully implemented yet"


class AnthropicClient(_UnavailableProviderClient):
    provider_label = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    sdk_import_name = "anthropic"
    sdk_error_message = (
        "Anthropic SDK is not installed. Anthropic real API call is not fully implemented yet"
    )
    not_implemented_message = "Anthropic real API call is not fully implemented yet"


class GeminiClient(_UnavailableProviderClient):
    provider_label = "gemini"
    api_key_env = "GEMINI_API_KEY"
    sdk_import_name = "google.generativeai"
    sdk_error_message = (
        "Gemini SDK is not installed. Gemini real API call is not fully implemented yet"
    )
    not_implemented_message = "Gemini real API call is not fully implemented yet"
