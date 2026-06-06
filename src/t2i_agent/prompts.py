from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "backend_category",
    "prompt_zh",
    "prompt_en",
    "evaluation_focus",
}


@dataclass(frozen=True)
class PromptRecord:
    id: str
    backend_category: str
    prompt_zh: str
    prompt_en: str
    evaluation_focus: str


def load_prompts(path: str | Path) -> dict[str, PromptRecord]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompts: dict[str, PromptRecord] = {}
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            missing = sorted(REQUIRED_FIELDS - payload.keys())
            if missing:
                raise ValueError(
                    f"{prompt_path}:{line_number} missing required fields: "
                    + ", ".join(missing)
                )
            record = PromptRecord(
                id=str(payload["id"]),
                backend_category=str(payload["backend_category"]),
                prompt_zh=str(payload["prompt_zh"]),
                prompt_en=str(payload["prompt_en"]),
                evaluation_focus=str(payload["evaluation_focus"]),
            )
            if record.id in prompts:
                raise ValueError(f"Duplicate prompt id: {record.id}")
            prompts[record.id] = record

    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts
