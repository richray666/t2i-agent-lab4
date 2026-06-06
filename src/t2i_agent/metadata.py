from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationRecord:
    backend: str
    repo_id: str
    prompt_id: str
    prompt: str
    seed: int
    steps: int
    cfg: float
    width: int
    height: int
    runtime_seconds: float
    offload: bool
    variant: str | None = None
    error: str | None = None


def save_generation_artifacts(
    image: Any,
    record: GenerationRecord,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(record)
    image_path = target_dir / f"{stem}.png"
    metadata_path = target_dir / f"{stem}.json"

    image.save(image_path)
    payload = asdict(record)
    payload["image_path"] = str(image_path)
    payload["metadata_path"] = str(metadata_path)
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return image_path, metadata_path


def save_error_record(
    output_dir: str | Path,
    *,
    backend: str,
    prompt_id: str,
    error: str,
    options: dict[str, Any],
) -> Path:
    target_dir = Path(output_dir) / "errors"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = target_dir / f"{timestamp}-{_slug(backend)}-{_slug(prompt_id)}.json"
    payload = {
        "backend": backend,
        "prompt_id": prompt_id,
        "error": error,
        "options": _jsonable(options),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _artifact_stem(record: GenerationRecord) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{timestamp}-{_slug(record.backend)}-{_slug(record.variant or 'default')}-"
        f"{_slug(record.prompt_id)}-seed{record.seed}"
    )


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "item"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
