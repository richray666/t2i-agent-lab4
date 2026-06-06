from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    name: str
    repo_id: str
    default_steps: int
    default_cfg: float
    default_width: int
    default_height: int
    default_aspect_ratio: str
    prompt_language: str
    pipeline_kind: str
    variant: str | None = None


def resolve_backend(backend: str, variant: str | None = None) -> BackendSpec:
    key = backend.lower()
    if key == "offline":
        if variant not in (None, "default", "demo"):
            raise ValueError("Offline backend only supports variant=None or 'demo'.")
        return BackendSpec(
            name="offline",
            repo_id="local/offline-demo",
            default_steps=4,
            default_cfg=1.0,
            default_width=768,
            default_height=768,
            default_aspect_ratio="1:1",
            prompt_language="zh",
            pipeline_kind="offline",
            variant="demo",
        )

    if key == "kolors":
        if variant not in (None, "default"):
            raise ValueError("Kolors only supports variant=None or variant='default'.")
        return BackendSpec(
            name="kolors",
            repo_id="Kwai-Kolors/Kolors-diffusers",
            default_steps=25,
            default_cfg=5.0,
            default_width=1024,
            default_height=1024,
            default_aspect_ratio="1:1",
            prompt_language="zh",
            pipeline_kind="kolors",
            variant="default",
        )

    if key == "lens":
        lens_variant = variant or "turbo"
        if lens_variant == "turbo":
            return BackendSpec(
                name="lens",
                repo_id="microsoft/Lens-Turbo",
                default_steps=4,
                default_cfg=1.0,
                default_width=1024,
                default_height=1024,
                default_aspect_ratio="1:1",
                prompt_language="en",
                pipeline_kind="lens",
                variant="turbo",
            )
        if lens_variant in ("full", "default"):
            return BackendSpec(
                name="lens",
                repo_id="microsoft/Lens",
                default_steps=20,
                default_cfg=5.0,
                default_width=1024,
                default_height=1024,
                default_aspect_ratio="1:1",
                prompt_language="en",
                pipeline_kind="lens",
                variant="full",
            )
        raise ValueError("Lens variant must be 'turbo' or 'full'.")

    raise ValueError("Backend must be 'offline', 'kolors', or 'lens'.")
