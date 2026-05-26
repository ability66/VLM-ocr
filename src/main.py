from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):  # type: ignore[no-redef]
        del desc
        return iterable

from src.decision import decide_consensus
from src.graph_fusion import FusedGraphResult, fuse_mermaid_outputs
from src.image_loader import load_image_tasks
from src.model_clients import (
    AnthropicClient,
    DashScopeClient,
    GeminiClient,
    MockVLMClient,
    OpenAICompatibleVLMClient,
    OpenAIClient,
)
from src.model_clients.base import BaseVLMClient
from src.normalizer import normalize_model_output
from src.prompt_builder import load_default_prompt
from src.schema import ConsensusResult, ModelOutput, ParsedLabel
from src.scorer import score_consensus
from src.validators import ValidationResult, validate_labels
from src.writer import (
    append_summary_record,
    build_summary_record,
    clear_previous_outputs,
    ensure_output_dirs,
    initialize_summary_file,
    write_image_result,
)

CLIENT_REGISTRY = {
    "mock": MockVLMClient,
    "dashscope": DashScopeClient,
    "qwen": DashScopeClient,
    "openai": OpenAIClient,
    "openai_compatible": OpenAICompatibleVLMClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-model image labeling pipeline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models-config", type=Path, default=Path("configs/models.yaml")
    )
    parser.add_argument(
        "--prompts-config", type=Path, default=Path("configs/prompts.yaml")
    )
    parser.add_argument("--mock", action="store_true", help="Only run mock models.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear previous per-image outputs before running.",
    )
    parser.add_argument("--concurrent-models", type=int, default=3)
    parser.add_argument("--concurrent-images", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retry", type=int, default=0)
    return parser.parse_args()


def load_model_configs(config_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        data = {"models": _fallback_models_parser(raw_text)}

    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError("Model config must contain a 'models' list")
    return [model for model in models if isinstance(model, dict)]


def _fallback_models_parser(raw_text: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_models = False

    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if not in_models:
            if stripped == "models:":
                in_models = True
            continue

        if stripped.startswith("- "):
            if current is not None:
                models.append(current)
            current = {}
            remainder = stripped[2:].strip()
            if remainder:
                key, value = _parse_key_value(remainder)
                current[key] = _parse_scalar(value)
            continue

        if current is None:
            continue

        key, value = _parse_key_value(stripped)
        current[key] = _parse_scalar(value)

    if current is not None:
        models.append(current)
    return models


def _parse_key_value(text: str) -> tuple[str, str]:
    key, separator, value = text.partition(":")
    if not separator:
        raise ValueError(f"Invalid YAML line: {text}")
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    normalized = value.strip()
    if normalized.lower() == "true":
        return True
    if normalized.lower() == "false":
        return False
    if normalized.startswith(("'", '"')) and normalized.endswith(("'", '"')):
        return normalized[1:-1]
    return normalized


def build_clients(model_configs: list[dict[str, Any]]) -> list[Any]:
    clients = []
    for model_config in model_configs:
        provider = str(model_config.get("provider", "")).strip().lower()
        model_name = str(model_config.get("name", "")).strip()
        if not provider or not model_name:
            print(f"Skipping invalid model config: {model_config}")
            continue

        client_class = CLIENT_REGISTRY.get(provider)
        if client_class is None:
            print(f"Skipping unsupported provider '{provider}' for model '{model_name}'")
            continue
        clients.append(client_class(model_name=model_name, config=model_config))
    return clients


def filter_enabled_models(
    model_configs: list[dict[str, Any]], mock_only: bool
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for config in model_configs:
        provider = str(config.get("provider", "")).strip().lower()
        if mock_only:
            if provider == "mock":
                filtered.append(config)
            continue
        if not bool(config.get("enabled", False)):
            continue
        filtered.append(config)
    return filtered


def apply_request_timeout_defaults(
    clients: list[BaseVLMClient], request_timeout: int
) -> None:
    for client in clients:
        timeout_value = client.config.get("timeout")
        if _parse_positive_int(timeout_value) is not None:
            continue
        client.config["timeout"] = request_timeout
        if hasattr(client, "timeout"):
            setattr(client, "timeout", request_timeout)


def call_model_with_retry(
    client: BaseVLMClient,
    image_task: Any,
    prompt: str,
    retry: int,
) -> ModelOutput:
    attempts = max(0, retry) + 1
    last_output: ModelOutput | None = None
    retries_used = 0

    for attempt in range(attempts):
        try:
            output = client.generate(image_task=image_task, prompt=prompt)
        except Exception as exc:
            output = ModelOutput(
                image_id=image_task.image_id,
                model_name=client.model_name,
                success=False,
                raw_text="",
                error=f"{type(exc).__name__}: {exc}",
            )
        last_output = output
        if output.success:
            return output

        error_text = str(output.error or "")
        if attempt >= attempts - 1 or not _is_retryable_error(error_text):
            break
        retries_used += 1
        sleep(2**attempt)

    if last_output is None:
        return ModelOutput(
            image_id=image_task.image_id,
            model_name=client.model_name,
            success=False,
            raw_text="",
            error=f"Model call failed before producing output after retries={retry}",
        )

    if retries_used > 0 and not last_output.success:
        final_output = last_output.model_copy(deep=True)
        message = str(final_output.error or "model call failed")
        final_output.error = f"{message} (retries={retries_used})"
        return final_output
    return last_output


def run_model_calls_for_image_concurrent(
    image_task: Any,
    clients: list[BaseVLMClient],
    prompt: str,
    max_workers: int,
    retry: int,
    request_timeout: int,
) -> list[ModelOutput]:
    if not clients:
        return []

    for client in clients:
        _apply_timeout_default_for_client(client, request_timeout)

    worker_count = max(1, min(max_workers, len(clients)))
    outputs: list[ModelOutput | None] = [None] * len(clients)

    if worker_count == 1:
        for index, client in enumerate(clients):
            outputs[index] = call_model_with_retry(
                client=client,
                image_task=image_task,
                prompt=prompt,
                retry=retry,
            )
        return [output for output in outputs if output is not None]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                call_model_with_retry,
                client,
                image_task,
                prompt,
                retry,
            ): index
            for index, client in enumerate(clients)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            client = clients[index]
            try:
                outputs[index] = future.result()
            except Exception as exc:
                outputs[index] = ModelOutput(
                    image_id=image_task.image_id,
                    model_name=client.model_name,
                    success=False,
                    raw_text="",
                    error=f"{type(exc).__name__}: {exc}",
                )

    finalized: list[ModelOutput] = []
    for index, output in enumerate(outputs):
        if output is not None:
            finalized.append(output)
            continue
        finalized.append(
            ModelOutput(
                image_id=image_task.image_id,
                model_name=clients[index].model_name,
                success=False,
                raw_text="",
                error="Model call returned no result",
            )
        )
    return finalized


def process_one_image(
    image_task: Any,
    clients: list[BaseVLMClient],
    prompt: str,
    output_dir: Path,
    concurrent_models: int,
    retry: int,
    request_timeout: int,
) -> dict[str, Any]:
    started_at = perf_counter()
    model_outputs: list[ModelOutput] = []
    normalized_results: list[ParsedLabel | None] = []
    graph_fusion_result: FusedGraphResult | None = None
    validation_result = ValidationResult(
        image_id=image_task.image_id,
        details={"label_count": 0, "evidence_text_pool": [], "per_label": []},
    )

    try:
        model_outputs = run_model_calls_for_image_concurrent(
            image_task=image_task,
            clients=clients,
            prompt=prompt,
            max_workers=concurrent_models,
            retry=retry,
            request_timeout=request_timeout,
        )
        normalized_outputs: list[ModelOutput] = []
        for output in model_outputs:
            normalized_output, normalized_label = normalize_model_output(output)
            normalized_outputs.append(normalized_output)
            normalized_results.append(normalized_label)
        model_outputs = normalized_outputs

        parsed_labels = [label for label in normalized_results if label is not None]
        parsed_outputs = [
            output
            for output, label in zip(model_outputs, normalized_results)
            if label is not None
        ]
        score_result = score_consensus(
            image_id=image_task.image_id,
            labels=parsed_labels,
            model_outputs=model_outputs,
        )
        if parsed_labels:
            validation_result = validate_labels(
                image_id=image_task.image_id,
                labels=parsed_labels,
            )
            evidence_texts = list(validation_result.details.get("evidence_text_pool", []))
            graph_fusion_result = fuse_mermaid_outputs(
                labels=parsed_labels,
                model_outputs=parsed_outputs,
                evidence_texts=evidence_texts,
            )
        else:
            validation_result = ValidationResult(
                image_id=image_task.image_id,
                details={"label_count": 0, "evidence_text_pool": [], "per_label": []},
            )
        consensus = decide_consensus(
            image_id=image_task.image_id,
            labels=parsed_labels,
            model_outputs=model_outputs,
            score_result=score_result,
            validation_result=validation_result,
            graph_fusion_result=graph_fusion_result,
        )
    except Exception as exc:
        validation_result = ValidationResult(
            image_id=image_task.image_id,
            warnings=[
                f"validation skipped because of pipeline error: {type(exc).__name__}: {exc}"
            ],
            details={
                "label_count": len([label for label in normalized_results if label is not None])
            },
        )
        consensus = ConsensusResult(
            image_id=image_task.image_id,
            type_agreement=0.0,
            caption_agreement=0.0,
            structure_agreement=0.0,
            overall_score=0.0,
            evidence_score=0.0,
            validator_score=0.0,
            hallucination_risk=0.0,
            accept_score=0.0,
            decision="failed",
            reasons=[f"Image pipeline error: {type(exc).__name__}: {exc}"],
            validation_errors=[],
            validation_warnings=list(validation_result.warnings),
            escalation_reasons=[],
        )

    record = write_image_result(
        output_dir=output_dir,
        image_task=image_task,
        model_outputs=model_outputs,
        normalized_results=normalized_results,
        consensus=consensus,
        validation_result=validation_result,
        graph_fusion_result=graph_fusion_result,
    )
    summary_record = build_summary_record(
        image_task=image_task,
        model_outputs=model_outputs,
        consensus=consensus,
        final_label=record["final_label"],
        final_label_status=record["final_label_status"],
        graph_fusion=record["graph_fusion"],
    )
    return {
        "image_id": image_task.image_id,
        "summary_record": summary_record,
        "consensus": consensus,
        "elapsed_seconds": round(perf_counter() - started_at, 3),
        "model_statuses": _format_model_statuses(model_outputs),
    }


def _format_model_statuses(model_outputs: list[ModelOutput]) -> list[str]:
    return [
        f"{output.model_name}:{'ok' if output.success else 'fail'}"
        for output in model_outputs
    ]


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _apply_timeout_default_for_client(client: BaseVLMClient, request_timeout: int) -> None:
    if _parse_positive_int(client.config.get("timeout")) is not None:
        return
    client.config["timeout"] = request_timeout
    if hasattr(client, "timeout"):
        setattr(client, "timeout", request_timeout)


def _is_retryable_error(error_text: str) -> bool:
    normalized = str(error_text or "").strip().lower()
    if not normalized:
        return True
    if any(marker in normalized for marker in ("api key", "missing", "unauthorized", "401")):
        return False
    return True


def main() -> int:
    args = parse_args()

    prompt = load_default_prompt(args.prompts_config)
    model_configs = load_model_configs(args.models_config)
    enabled_configs = filter_enabled_models(model_configs=model_configs, mock_only=args.mock)
    clients = build_clients(enabled_configs)
    apply_request_timeout_defaults(clients=clients, request_timeout=args.request_timeout)

    if not clients:
        print("No enabled models found for the current run")
        return 0

    image_tasks = load_image_tasks(args.data_dir)
    if args.limit is not None:
        image_tasks = image_tasks[: args.limit]

    if not image_tasks:
        print("No images found in data directory")
        return 0

    ensure_output_dirs(args.output_dir)
    if args.overwrite:
        clear_previous_outputs(args.output_dir)
    summary_path = initialize_summary_file(args.output_dir)

    concurrent_models = max(1, args.concurrent_models)
    concurrent_images = max(1, args.concurrent_images)
    retry = max(0, args.retry)

    print(f"enabled models: {len(clients)}")
    print(f"model names: {[client.model_name for client in clients]}")
    print(f"concurrent_models: {concurrent_models}")
    print(f"concurrent_images: {concurrent_images}")
    print(f"retry: {retry}")
    print(f"request_timeout: {args.request_timeout}")

    results_by_index: dict[int, dict[str, Any]] = {}
    if concurrent_images == 1:
        for index, image_task in enumerate(tqdm(image_tasks, desc="Processing images")):
            result = process_one_image(
                image_task=image_task,
                clients=clients,
                prompt=prompt,
                output_dir=args.output_dir,
                concurrent_models=concurrent_models,
                retry=retry,
                request_timeout=args.request_timeout,
            )
            results_by_index[index] = result
            print(
                f"completed {image_task.file_name} in {result['elapsed_seconds']:.3f}s | "
                + ", ".join(result["model_statuses"])
            )
    else:
        with ThreadPoolExecutor(max_workers=concurrent_images) as executor:
            future_to_index = {
                executor.submit(
                    process_one_image,
                    image_task,
                    clients,
                    prompt,
                    args.output_dir,
                    concurrent_models,
                    retry,
                    args.request_timeout,
                ): (index, image_task.file_name)
                for index, image_task in enumerate(image_tasks)
            }
            for future in tqdm(as_completed(future_to_index), desc="Processing images"):
                index, file_name = future_to_index[future]
                result = future.result()
                results_by_index[index] = result
                print(
                    f"completed {file_name} in {result['elapsed_seconds']:.3f}s | "
                    + ", ".join(result["model_statuses"])
                )

    stats = {"accepted": 0, "review": 0, "failed": 0}
    for index in sorted(results_by_index):
        result = results_by_index[index]
        append_summary_record(summary_path, result["summary_record"])
        consensus = result["consensus"]
        stats[consensus.decision] += 1

    total_images = len(image_tasks)
    print(f"total images: {total_images}")
    print(f"accepted: {stats['accepted']}")
    print(f"review: {stats['review']}")
    print(f"failed: {stats['failed']}")
    print(f"output path: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
