from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.model_clients.base import BaseVLMClient
from src.schema import ImageTask


class MockVLMClient(BaseVLMClient):
    def _generate_impl(self, image_task: ImageTask, prompt: str) -> dict[str, Any]:
        del prompt
        image_type = self._infer_image_type(image_task)
        payload = self._build_payload(image_task=image_task, image_type=image_type)
        return {
            "success": True,
            "raw_text": self._render_payload(image_task=image_task, payload=payload),
            "error": None,
        }

    def _infer_image_type(self, image_task: ImageTask) -> str:
        path = Path(image_task.image_path)
        haystack = " ".join([part.lower() for part in path.parts])

        if "flow" in haystack:
            return "flowchart"
        if "chart" in haystack or "plot" in haystack or "graph" in haystack:
            return "chart"
        if "table" in haystack or "tabular" in haystack:
            return "table"
        if "screenshot" in haystack or "screen" in haystack:
            return "screenshot"
        if "document" in haystack or "doc" in haystack or "page" in haystack:
            return "document"
        if "diagram" in haystack:
            return "diagram"
        if "mixed" in haystack:
            return "mixed"
        if "unknown" in haystack:
            return "unknown"
        return "natural_image"

    def _build_payload(self, image_task: ImageTask, image_type: str) -> dict[str, Any]:
        visible_text = self._visible_text(image_type=image_type, image_task=image_task)
        caption = self._caption(image_type=image_type)
        payload: dict[str, Any] = {
            "image_type": image_type,
            "caption": caption,
            "caption_structured": self._caption_structured(
                image_type=image_type,
                caption=caption,
                visible_text=visible_text,
            ),
            "structured_label": self._structured_label(image_type=image_type, variant="base"),
            "visible_text": visible_text,
            "uncertainty": "仅依据图像可见区域生成，局部文字可能不完整。",
            "warnings": [],
        }

        if self.model_name == "mock_vlm_b":
            payload["structured_label"] = self._structured_label(
                image_type=image_type, variant="b"
            )
            payload["warnings"] = ["模型输出经过代码块包裹。"]
        elif self.model_name == "mock_vlm_c":
            payload["structured_label"] = self._structured_label(
                image_type=image_type, variant="c"
            )
            payload.pop("uncertainty", None)
            if image_type in {"table", "chart"}:
                payload.pop("visible_text", None)
            if image_type == "unknown":
                payload.pop("structured_label", None)
            if image_type == "flowchart":
                payload["warnings"] = ["流程图节点文字可能存在截断。"]
        return payload

    def _render_payload(self, image_task: ImageTask, payload: dict[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if self.model_name == "mock_vlm_b":
            return f"以下是识别结果：\n```json\n{text}\n```"
        if self.model_name == "mock_vlm_c" and image_task.file_name.endswith("4.png"):
            return f"```{text}```"
        return text

    def _caption(self, image_type: str) -> str:
        mapping = {
            "natural_image": "自然图像，含可见主体与场景。",
            "chart": "图表，含坐标与数据标注。",
            "table": "表格图像，含行列内容。",
            "flowchart": "流程图，含步骤与连线。",
            "document": "文档图像，含标题与正文。",
            "screenshot": "界面截图，含控件与文字。",
            "diagram": "示意图，含结构与标注。",
            "mixed": "混合图像，含图形与文字。",
            "unknown": "图像内容复杂，类型不稳定。",
        }
        return mapping.get(image_type, mapping["unknown"])

    def _caption_structured(
        self,
        image_type: str,
        caption: str,
        visible_text: list[str],
    ) -> dict[str, Any]:
        main_subject_mapping = {
            "natural_image": "",
            "chart": "图表区域",
            "table": "表格区域",
            "flowchart": "流程步骤",
            "document": "文档版面",
            "screenshot": "界面区域",
            "diagram": "关系结构",
            "mixed": "",
            "unknown": "",
        }
        structure_summary_mapping = {
            "natural_image": "可见主体与场景内容。",
            "chart": "可见图表结构与数据标注。",
            "table": "可见表格结构与单元格内容。",
            "flowchart": "可见流程节点、连线与分支关系。",
            "document": "可见文档版面与文字块结构。",
            "screenshot": "可见界面布局与控件区域。",
            "diagram": "可见示意结构与关系标注。",
            "mixed": "可见图文混合内容与局部结构。",
            "unknown": "可见图像内容，但结构类型不稳定。",
        }
        return {
            "brief": caption,
            "visual_type": image_type,
            "main_subject": main_subject_mapping.get(image_type, ""),
            "visible_title": "",
            "key_visible_text": visible_text[:10],
            "structure_summary": structure_summary_mapping.get(
                image_type, structure_summary_mapping["unknown"]
            ),
            "caption_source": "generated",
            "confidence": "medium",
        }

    def _structured_label(self, image_type: str, variant: str) -> dict[str, str]:
        base_mapping = {
            "natural_image": {"kind": "none", "content": "", "format": "none"},
            "chart": {
                "kind": "table",
                "content": "| 项目 | 数值 |\n| --- | --- |\n| 类别A | 上升 |\n| 类别B | 下降 |",
                "format": "markdown",
            },
            "table": {
                "kind": "table",
                "content": "| 列1 | 列2 |\n| --- | --- |\n| 单元格A | 单元格B |\n| 单元格C | 单元格D |",
                "format": "markdown",
            },
            "flowchart": {
                "kind": "mermaid",
                "content": "flowchart TD\nA[开始] --> B[处理]\nB --> C[结束]",
                "format": "mermaid",
            },
            "document": {
                "kind": "text",
                "content": "文档图，包含标题、正文段落和若干列表区域。",
                "format": "plain_text",
            },
            "screenshot": {
                "kind": "text",
                "content": "截图，包含顶部栏、主体区域和按钮或输入框。",
                "format": "plain_text",
            },
            "diagram": {
                "kind": "text",
                "content": "示意图，包含多个标注对象及它们之间的关系。",
                "format": "plain_text",
            },
            "mixed": {
                "kind": "text",
                "content": "混合图像，包含图形、表格或说明文字。",
                "format": "plain_text",
            },
            "unknown": {
                "kind": "text",
                "content": "图像结构不明确，暂无法可靠结构化。",
                "format": "plain_text",
            },
        }
        value = dict(base_mapping.get(image_type, base_mapping["unknown"]))

        if variant == "b":
            if image_type == "flowchart":
                value["content"] = "flowchart TD\nA[开始] --> B[步骤]\nB --> C[结果]"
            elif image_type in {"chart", "table"}:
                value["kind"] = "grid"
                value.pop("format", None)
            elif image_type == "natural_image":
                value["kind"] = "text"
                value["content"] = "自然图像，不适合进一步表格化。"
                value["format"] = "plain_text"
        elif variant == "c":
            if image_type == "flowchart":
                value["content"] = "flowchart TD\nA[开始] --> B[处理]\nB --> D[复核]\nD --> C[结束]"
                value.pop("format", None)
            elif image_type == "table":
                value["content"] = "列1,列2\n单元格A,单元格B\n单元格C,单元格D"
                value["format"] = "csv"
            elif image_type == "chart":
                value["kind"] = "text"
                value["content"] = "图表，难以准确还原全部数据。"
                value["format"] = "plain_text"
        return value

    def _visible_text(self, image_type: str, image_task: ImageTask) -> list[str]:
        base = [image_task.file_name]
        mapping = {
            "chart": ["标题", "横轴", "纵轴"],
            "table": ["表头", "行标签", "列标签"],
            "flowchart": ["开始", "处理", "结束"],
            "document": ["标题", "段落"],
            "screenshot": ["菜单", "按钮"],
            "diagram": ["标注", "箭头"],
        }
        return base + mapping.get(image_type, [])
