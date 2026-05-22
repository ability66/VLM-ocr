from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, console, ensure_dir, load_yaml, read_text, relative_path, write_json, write_text
from flowchart_graph_common import (
    FLOWCHART_CROPS_DIR,
    FLOWCHART_MANUAL_MERMAID_DIR,
    FLOWCHART_RAW_DIR,
    build_crop_records,
    ensure_flowchart_dirs,
    extract_json_candidate,
    image_data_uri,
    mermaid_from_graph,
    normalize_graph_payload,
    parse_mermaid_flowchart,
    write_run_status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run flowchart-to-graph tools on crop images.")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/flowchart_crops"), help="Directory containing flowchart crop images.")
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/flowchart_graph/raw"), help="Directory for raw tool outputs.")
    parser.add_argument("--tools-config", type=Path, default=Path("configs/tools.yaml"), help="Tool config YAML path.")
    parser.add_argument("--tools", type=str, default="", help="Optional comma-separated tool list.")
    parser.add_argument("--crop-id", type=str, default="", help="Optional single crop_id filter for smoke tests.")
    return parser


def selected_tools(config: dict[str, Any], requested: str) -> list[str]:
    if requested.strip():
        items = [item.strip() for item in requested.split(",") if item.strip()]
        return [item for item in items if item in config]
    return list(config.keys())


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Diagram2GraphHFRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self.process_vision_info = None
        self.prepare_error: str | None = None
        self.torch = None
        self.model_dtype_name = ""
        self.device_name = ""

    def choose_torch_dtype(self, torch_module: Any) -> Any:
        if not torch_module.cuda.is_available():
            self.model_dtype_name = "cpu"
            return getattr(torch_module, "float32")
        capability = torch_module.cuda.get_device_capability(0)
        self.device_name = torch_module.cuda.get_device_name(0)
        if capability[0] >= 8 and hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
            self.model_dtype_name = "bfloat16"
            return getattr(torch_module, "bfloat16")
        self.model_dtype_name = "float16"
        return getattr(torch_module, "float16")

    def build_graph_artifacts(self, parsed: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, list[str]]:
        parse_errors: list[str] = []
        if isinstance(parsed, dict):
            if isinstance(parsed.get("result"), dict):
                parsed = parsed["result"]
            elif isinstance(parsed.get("graph"), dict):
                parsed = parsed["graph"]

        if not isinstance(parsed, dict):
            raise ValueError("模型输出 JSON 顶层不是 object，无法抽取 nodes/edges/mermaid。")

        raw_nodes = parsed.get("nodes") if isinstance(parsed.get("nodes"), list) else []
        raw_edges = parsed.get("edges") if isinstance(parsed.get("edges"), list) else []
        label_map: dict[str, str] = {}
        normalized_edges: list[dict[str, Any]] = []
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue
            source_label = str(edge.get("source_label", "") or edge.get("source_text", "") or "").strip()
            target_label = str(edge.get("target_label", "") or edge.get("target_text", "") or "").strip()
            if source_label:
                label_map[source] = source_label
            if target_label:
                label_map[target] = target_label
            normalized_edges.append(
                {
                    "source": source,
                    "target": target,
                    "label": str(edge.get("label", "") or edge.get("relationship_value", "") or "").strip(),
                    "direction": "forward",
                    "raw_edge": edge,
                }
            )

        normalized_nodes: list[dict[str, Any]] = []
        for node in raw_nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            text = (
                str(node.get("text", "") or node.get("label", "") or node.get("node_text", "") or label_map.get(node_id, "")).strip()
                or node_id
            )
            shape = str(node.get("shape", "") or node.get("type_of_node", "") or "rect").strip() or "rect"
            normalized_nodes.append(
                {
                    "id": node_id,
                    "text": text,
                    "shape": shape,
                    "bbox": node.get("bbox"),
                    "raw_node": node,
                }
            )

        nodes, edges, mermaid = normalize_graph_payload(
            {
                "nodes": normalized_nodes,
                "edges": normalized_edges,
                "mermaid": parsed.get("mermaid", ""),
            }
        )
        if not mermaid and isinstance(parsed.get("mermaid_code"), str):
            mermaid = parsed["mermaid_code"].strip()
        if mermaid and (not nodes and not edges):
            nodes, edges, mermaid_errors = parse_mermaid_flowchart(mermaid)
            parse_errors.extend(mermaid_errors)
        if not mermaid and (nodes or edges):
            mermaid = mermaid_from_graph(nodes, edges)
        return nodes, edges, mermaid, parse_errors

    def looks_truncated_json(self, text: str, exc: Exception) -> bool:
        message = str(exc).lower()
        if "unterminated string" in message or "expecting value" in message:
            return True
        return text.count("{") != text.count("}") or text.count("[") != text.count("]")

    def generate_text(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.as_posix()},
                    {"type": "text", "text": "Return only JSON."},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def failure_suggestion(self, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "outofmemory" in lowered or "cuda out of memory" in lowered:
            return "尝试减小输入分辨率、限制 max_pixels，或换更大显存卡。"
        if "no module named" in lowered:
            return "先安装缺失依赖，然后重新执行单图 smoke test。"
        if "flash_attn" in lowered or "sdpa" in lowered:
            return "尝试升级 transformers，或显式关闭相关高性能 attention 后端。"
        if "json" in lowered:
            return "模型已输出文本，但 JSON 结构不稳定；建议收紧 prompt 或增加后处理容错。"
        return "先查看 raw_output.txt 和错误堆栈，确认是模型加载失败还是推理阶段失败。"

    def prepare(self) -> str | None:
        if self.prepare_error is not None:
            return self.prepare_error
        if self.model is not None and self.processor is not None and self.process_vision_info is not None:
            return None
        if not self.config.get("enabled", True):
            self.prepare_error = "diagram2graph_hf is disabled in configs/tools.yaml"
            return self.prepare_error
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        except Exception as exc:
            self.prepare_error = f"依赖不可用: {exc}"
            return self.prepare_error
        self.torch = torch
        if not torch.cuda.is_available():
            self.prepare_error = "GPU 不可用，跳过 diagram2graph_hf。"
            return self.prepare_error
        try:
            model_id = str(self.config.get("model_id", "zackriya/diagram2graph"))
            torch_dtype = self.choose_torch_dtype(torch)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch_dtype,
            )
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                model_id,
                min_pixels=int(self.config.get("min_pixels", 256 * 28 * 28)),
                max_pixels=int(self.config.get("max_pixels", 1280 * 28 * 28)),
            )
            self.process_vision_info = process_vision_info
            return None
        except Exception as exc:
            self.prepare_error = f"模型加载失败: {exc}"
            return self.prepare_error

    def run(self, crop: dict[str, str], out_dir: Path) -> None:
        if not self.config.get("enabled", True):
            write_run_status(out_dir, tool="diagram2graph_hf", crop_id=crop["crop_id"], status="skipped", reason="disabled")
            return
        reason = self.prepare()
        if reason:
            write_run_status(out_dir, tool="diagram2graph_hf", crop_id=crop["crop_id"], status="skipped", reason=reason)
            return

        prompt = str(
            self.config.get(
                "prompt",
                """
Extract the flowchart into valid JSON only.
Use exactly this schema:
{
  "nodes": [{"id":"","text":"","shape":"","bbox":null}],
  "edges": [{"source":"","target":"","label":"","direction":""}],
  "mermaid": ""
}
Do not add explanation or markdown fences.
""".strip(),
            )
        )
        image_path = (PROJECT_ROOT / crop["image_path"]).resolve()
        try:
            initial_max_new_tokens = int(self.config.get("max_new_tokens", 4096))
            retry_max_new_tokens = int(self.config.get("retry_max_new_tokens", max(initial_max_new_tokens, 4096)))
            output_text = self.generate_text(image_path, prompt, initial_max_new_tokens)
            write_text(out_dir / "raw_output.txt", output_text)
            try:
                parsed = json.loads(extract_json_candidate(output_text))
            except json.JSONDecodeError as exc:
                if self.looks_truncated_json(output_text, exc) and retry_max_new_tokens > initial_max_new_tokens:
                    write_text(out_dir / "raw_output_attempt1.txt", output_text)
                    output_text = self.generate_text(image_path, prompt, retry_max_new_tokens)
                    write_text(out_dir / "raw_output.txt", output_text)
                parsed = json.loads(extract_json_candidate(output_text))
            write_json(out_dir / "raw_output.json", parsed)
            nodes, edges, mermaid, parse_errors = self.build_graph_artifacts(parsed)
            write_json(out_dir / "nodes.json", nodes)
            write_json(out_dir / "edges.json", edges)
            write_text(out_dir / "mermaid.mmd", (mermaid.strip() + "\n") if mermaid.strip() else "")
            write_json(
                out_dir / "parsed_graph.json",
                {
                    "nodes": nodes,
                    "edges": edges,
                    "mermaid": mermaid,
                    "parse_errors": parse_errors,
                },
            )
            write_run_status(
                out_dir,
                tool="diagram2graph_hf",
                crop_id=crop["crop_id"],
                status="success",
                details={
                    "model_id": self.config.get("model_id", "zackriya/diagram2graph"),
                    "device_name": self.device_name,
                    "torch_dtype": self.model_dtype_name,
                    "max_new_tokens": retry_max_new_tokens if (out_dir / "raw_output_attempt1.txt").exists() else initial_max_new_tokens,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "mermaid_generated": bool(mermaid.strip()),
                    "parse_errors": parse_errors,
                },
            )
        except Exception as exc:
            tb = traceback.format_exc()
            write_text(out_dir / "error.txt", tb)
            write_run_status(
                out_dir,
                tool="diagram2graph_hf",
                crop_id=crop["crop_id"],
                status="failed",
                reason=str(exc),
                details={
                    "device_name": self.device_name,
                    "torch_dtype": self.model_dtype_name,
                    "traceback": tb,
                    "next_step": self.failure_suggestion(exc),
                },
            )


class MinerUProRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.client = None
        self.prepare_error: str | None = None

    def prepare(self) -> str | None:
        if self.prepare_error is not None:
            return self.prepare_error
        if self.client is not None:
            return None
        if not self.config.get("enabled", False):
            self.prepare_error = "mineru_pro_image_analysis is disabled in configs/tools.yaml"
            return self.prepare_error
        server_url = str(self.config.get("server_url", "")).strip()
        if not server_url:
            self.prepare_error = "未配置 server_url；当前适配器需要可用的 MinerU Pro 服务。"
            return self.prepare_error
        try:
            from mineru_vl_utils import MinerUClient
        except Exception as exc:
            self.prepare_error = f"mineru_vl_utils 不可用: {exc}"
            return self.prepare_error
        try:
            self.client = MinerUClient(backend="http-client", server_url=server_url)
            return None
        except Exception as exc:
            self.prepare_error = f"MinerUClient 初始化失败: {exc}"
            return self.prepare_error

    def blocks_to_markdown(self, blocks: Any) -> str:
        if not isinstance(blocks, list):
            return ""
        lines: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or block.get("content") or "").strip()
            block_type = str(block.get("type") or block.get("sub_type") or "block").strip()
            if text:
                lines.append(f"- [{block_type}] {text}")
        return "\n".join(lines)

    def run(self, crop: dict[str, str], out_dir: Path) -> None:
        if not self.config.get("enabled", False):
            write_run_status(out_dir, tool="mineru_pro_image_analysis", crop_id=crop["crop_id"], status="skipped", reason="disabled")
            return
        reason = self.prepare()
        if reason:
            write_run_status(out_dir, tool="mineru_pro_image_analysis", crop_id=crop["crop_id"], status="skipped", reason=reason)
            return
        try:
            from PIL import Image
        except Exception as exc:
            write_run_status(out_dir, tool="mineru_pro_image_analysis", crop_id=crop["crop_id"], status="skipped", reason=f"Pillow 不可用: {exc}")
            return

        image_path = (PROJECT_ROOT / crop["image_path"]).resolve()
        try:
            with Image.open(image_path) as image:
                blocks = self.client.two_step_extract(image)
            write_json(out_dir / "content_list.json", blocks)
            markdown = self.blocks_to_markdown(blocks)
            if markdown:
                write_text(out_dir / "raw_output.md", markdown)
            write_text(out_dir / "raw_output.txt", json.dumps(blocks, ensure_ascii=False, indent=2))
            write_run_status(
                out_dir,
                tool="mineru_pro_image_analysis",
                crop_id=crop["crop_id"],
                status="success",
                details={"backend": "http-client", "image_analysis": True},
            )
        except Exception as exc:
            write_run_status(out_dir, tool="mineru_pro_image_analysis", crop_id=crop["crop_id"], status="failed", reason=str(exc))


def run_flowchart2mermaid_manual(config: dict[str, Any], crop: dict[str, str], out_dir: Path) -> None:
    if not config.get("enabled", True):
        write_run_status(out_dir, tool="flowchart2mermaid_manual", crop_id=crop["crop_id"], status="skipped", reason="disabled")
        return
    manual_dir = (PROJECT_ROOT / config.get("manual_dir", FLOWCHART_MANUAL_MERMAID_DIR.as_posix())).resolve()
    image_stem = Path(crop["image_path"]).stem
    candidates = [
        manual_dir / f"{crop['crop_id']}.mmd",
        manual_dir / f"{image_stem}.mmd",
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        write_run_status(
            out_dir,
            tool="flowchart2mermaid_manual",
            crop_id=crop["crop_id"],
            status="skipped",
            reason=f"未找到手工 Mermaid 文件，期望位置: {relative_path(candidates[0], PROJECT_ROOT)}",
        )
        return
    content = read_text(source_path).strip()
    if not content:
        write_run_status(out_dir, tool="flowchart2mermaid_manual", crop_id=crop["crop_id"], status="failed", reason="Mermaid 文件为空")
        return
    write_text(out_dir / "raw_output.mmd", content + "\n")
    write_run_status(
        out_dir,
        tool="flowchart2mermaid_manual",
        crop_id=crop["crop_id"],
        status="success",
        details={"source_mmd_path": relative_path(source_path, PROJECT_ROOT)},
    )


def call_json_api(url: str, headers: dict[str, str], payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def run_vlm_json_mermaid(config: dict[str, Any], crop: dict[str, str], out_dir: Path) -> None:
    if not config.get("enabled", False):
        write_run_status(out_dir, tool="vlm_json_mermaid", crop_id=crop["crop_id"], status="skipped", reason="disabled")
        return

    provider = str(config.get("provider", "openai_compatible")).strip()
    model = str(config.get("model", "")).strip()
    base_url = str(config.get("base_url", "")).strip()
    api_key_env = str(config.get("api_key_env", "")).strip()
    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
    if not model:
        write_run_status(out_dir, tool="vlm_json_mermaid", crop_id=crop["crop_id"], status="skipped", reason="未配置 model")
        return
    if not api_key:
        write_run_status(out_dir, tool="vlm_json_mermaid", crop_id=crop["crop_id"], status="skipped", reason=f"环境变量 {api_key_env or 'API_KEY'} 未配置")
        return

    prompt = str(
        config.get(
            "prompt",
            """
Return only valid JSON:
{
  "nodes": [{"id":"","text":"","shape":"","bbox":null}],
  "edges": [{"source":"","target":"","label":"","direction":""}],
  "mermaid": ""
}
""".strip(),
        )
    )
    image_path = (PROJECT_ROOT / crop["image_path"]).resolve()
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/png"

    try:
        if provider == "google_gemini":
            endpoint = base_url.strip() or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            payload = {
                "systemInstruction": {"parts": [{"text": prompt}]},
                "contents": [
                    {
                        "parts": [
                            {"text": "Analyze this flowchart image and return only JSON."},
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": image_base64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": config.get("temperature", 0),
                    "responseMimeType": "application/json",
                },
            }
            response = call_json_api(
                endpoint,
                {
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                payload,
            )
            write_json(out_dir / "raw_response.json", response)
            candidates = response.get("candidates", [])
            parts = []
            if candidates:
                candidate_parts = candidates[0].get("content", {}).get("parts", [])
                parts = [part.get("text", "") for part in candidate_parts if isinstance(part, dict)]
            output_text = "\n".join(part for part in parts if part).strip()
            write_text(out_dir / "raw_output.txt", output_text)
            try:
                parsed = json.loads(extract_json_candidate(output_text))
                write_json(out_dir / "raw_output.json", parsed)
            except Exception:
                pass
            write_run_status(
                out_dir,
                tool="vlm_json_mermaid",
                crop_id=crop["crop_id"],
                status="success",
                details={"provider": provider, "model": model},
            )
        else:
            if not base_url:
                write_run_status(out_dir, tool="vlm_json_mermaid", crop_id=crop["crop_id"], status="skipped", reason="未配置 base_url")
                return
            endpoint = base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": model,
                "temperature": config.get("temperature", 0),
                "max_tokens": config.get("max_tokens", 2048),
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this flowchart image and return only JSON."},
                            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
                        ],
                    },
                ],
            }
            response = call_json_api(
                endpoint,
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                payload,
            )
            write_json(out_dir / "raw_response.json", response)
            message = response["choices"][0]["message"]["content"]
            if isinstance(message, list):
                output_text = "\n".join(
                    item.get("text", "") for item in message if isinstance(item, dict)
                ).strip()
            else:
                output_text = str(message).strip()
            write_text(out_dir / "raw_output.txt", output_text)
            try:
                parsed = json.loads(extract_json_candidate(output_text))
                write_json(out_dir / "raw_output.json", parsed)
            except Exception:
                pass
            write_run_status(
                out_dir,
                tool="vlm_json_mermaid",
                crop_id=crop["crop_id"],
                status="success",
                details={"provider": provider, "model": model},
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        write_text(out_dir / "stderr.txt", body)
        write_run_status(out_dir, tool="vlm_json_mermaid", crop_id=crop["crop_id"], status="failed", reason=f"HTTP {exc.code}: {body}")
    except Exception as exc:
        write_run_status(out_dir, tool="vlm_json_mermaid", crop_id=crop["crop_id"], status="failed", reason=str(exc))


def run_flowextract_external(config: dict[str, Any], crop: dict[str, str], out_dir: Path) -> None:
    if not config.get("enabled", False):
        write_run_status(out_dir, tool="flowextract_external", crop_id=crop["crop_id"], status="skipped", reason="disabled")
        return
    command_template = str(config.get("command", "")).strip()
    if not command_template:
        write_run_status(out_dir, tool="flowextract_external", crop_id=crop["crop_id"], status="skipped", reason="未配置 command")
        return

    image_path = (PROJECT_ROOT / crop["image_path"]).resolve()
    command = command_template.format(
        image_path=image_path.as_posix(),
        crop_id=crop["crop_id"],
        out_dir=out_dir.as_posix(),
        image_name=image_path.name,
    )
    write_text(out_dir / "command.txt", command + "\n")
    completed = subprocess.run(
        command,
        shell=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    write_text(out_dir / "stdout.txt", completed.stdout)
    write_text(out_dir / "stderr.txt", completed.stderr)
    if completed.returncode == 0:
        write_run_status(out_dir, tool="flowextract_external", crop_id=crop["crop_id"], status="success", details={"command": command})
    else:
        write_run_status(
            out_dir,
            tool="flowextract_external",
            crop_id=crop["crop_id"],
            status="failed",
            reason=f"command exited with code {completed.returncode}",
            details={"command": command},
        )


def main() -> None:
    args = build_parser().parse_args()
    ensure_flowchart_dirs()
    crops_dir = (PROJECT_ROOT / args.crops_dir).resolve()
    raw_dir = (PROJECT_ROOT / args.raw_dir).resolve()
    config = load_yaml((PROJECT_ROOT / args.tools_config).resolve())
    crop_records = build_crop_records(crops_dir)
    if args.crop_id.strip():
        crop_records = [crop for crop in crop_records if crop["crop_id"] == args.crop_id.strip()]

    if not crop_records:
        console.print("[yellow]请先放入流程图 crop 图片。[/yellow]")
        return

    tools = selected_tools(config, args.tools)
    diagram_runner = Diagram2GraphHFRunner(config.get("diagram2graph_hf", {}))
    mineru_runner = MinerUProRunner(config.get("mineru_pro_image_analysis", {}))

    for tool in tools:
        tool_config = config.get(tool, {})
        for crop in crop_records:
            out_dir = reset_dir(raw_dir / tool / crop["crop_id"])
            try:
                if tool == "flowchart2mermaid_manual":
                    run_flowchart2mermaid_manual(tool_config, crop, out_dir)
                elif tool == "diagram2graph_hf":
                    diagram_runner.run(crop, out_dir)
                elif tool == "mineru_pro_image_analysis":
                    mineru_runner.run(crop, out_dir)
                elif tool == "vlm_json_mermaid":
                    run_vlm_json_mermaid(tool_config, crop, out_dir)
                elif tool == "flowextract_external":
                    run_flowextract_external(tool_config, crop, out_dir)
                else:
                    write_run_status(out_dir, tool=tool, crop_id=crop["crop_id"], status="skipped", reason="unknown tool")
            except Exception as exc:
                write_run_status(out_dir, tool=tool, crop_id=crop["crop_id"], status="failed", reason=str(exc))

    console.print(f"[green]已写入 flowchart raw 结果目录：[/green]{relative_path(raw_dir, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
