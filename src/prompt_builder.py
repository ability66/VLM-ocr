from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_default_prompt(config_path: Path) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        data = _fallback_prompt_parser(raw_text)

    prompt = data.get("default_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing non-empty 'default_prompt' in prompts config")

    return prompt.strip()


def _fallback_prompt_parser(raw_text: str) -> dict[str, str]:
    lines = raw_text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("default_prompt:"):
            continue

        if stripped.endswith("|"):
            block: list[str] = []
            for subline in lines[index + 1 :]:
                if subline.startswith("  "):
                    block.append(subline[2:])
                    continue
                if not subline.strip():
                    block.append("")
                    continue
                break
            return {"default_prompt": "\n".join(block).rstrip()}

        _, _, value = stripped.partition(":")
        return {"default_prompt": value.strip().strip("'").strip('"')}

    raise ValueError("Missing 'default_prompt' in prompts config")
