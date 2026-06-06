from __future__ import annotations

import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .backends import BackendSpec, resolve_backend
from .metadata import GenerationRecord, save_generation_artifacts
from .prompts import PromptRecord, load_prompts


@dataclass(frozen=True)
class GenerateOptions:
    backend: str
    prompt_id: str
    prompt_file: Path
    output_dir: Path
    variant: str | None = None
    seed: int = 0
    steps: int | None = None
    cfg: float | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    language: str | None = None
    offload: bool = False
    dtype: str | None = None


def generate(options: GenerateOptions) -> tuple[Path, Path]:
    spec = resolve_backend(options.backend, options.variant)
    prompts = load_prompts(options.prompt_file)
    if options.prompt_id not in prompts:
        raise ValueError(f"Prompt id not found: {options.prompt_id}")
    prompt_record = prompts[options.prompt_id]
    prompt = select_prompt(prompt_record, options.language or spec.prompt_language)

    steps = options.steps or spec.default_steps
    cfg = options.cfg if options.cfg is not None else spec.default_cfg
    width = options.width or spec.default_width
    height = options.height or spec.default_height
    aspect_ratio = options.aspect_ratio or spec.default_aspect_ratio

    start = time.perf_counter()
    image = _generate_image(
        spec=spec,
        prompt_id=prompt_record.id,
        prompt=prompt,
        seed=options.seed,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        offload=options.offload,
        dtype_name=options.dtype,
    )
    runtime = round(time.perf_counter() - start, 3)

    record = GenerationRecord(
        backend=spec.name,
        repo_id=spec.repo_id,
        prompt_id=prompt_record.id,
        prompt=prompt,
        seed=options.seed,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
        runtime_seconds=runtime,
        offload=options.offload,
        variant=spec.variant,
    )
    return save_generation_artifacts(image, record, options.output_dir)


def select_prompt(record: PromptRecord, language: str) -> str:
    if language == "zh":
        return record.prompt_zh
    if language == "en":
        return record.prompt_en
    raise ValueError("Language must be 'zh' or 'en'.")


def _generate_image(
    *,
    spec: BackendSpec,
    prompt_id: str,
    prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    aspect_ratio: str,
    offload: bool,
    dtype_name: str | None,
):
    if spec.pipeline_kind == "offline":
        return _generate_offline(
            spec=spec,
            prompt_id=prompt_id,
            prompt=prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
        )
    if spec.pipeline_kind == "kolors":
        return _generate_kolors(
            spec=spec,
            prompt=prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            offload=offload,
            dtype_name=dtype_name,
        )
    if spec.pipeline_kind == "lens":
        return _generate_lens(
            spec=spec,
            prompt=prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            aspect_ratio=aspect_ratio,
            offload=offload,
            dtype_name=dtype_name,
        )
    raise ValueError(f"Unsupported pipeline kind: {spec.pipeline_kind}")


def _generate_offline(
    *,
    spec: BackendSpec,
    prompt_id: str,
    prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
):
    """Produce a deterministic placeholder image without downloading any model.

    This backend lets the agent run end-to-end on machines that cannot fit the
    17GB/29GB real checkpoints. The image is reproducible for a given
    (prompt, seed) pair, so it still demonstrates the full pipeline: prompt
    selection, sampling parameters, image creation, and metadata recording.
    """
    import colorsys
    import hashlib
    import random

    from PIL import Image, ImageDraw

    digest = hashlib.sha256(f"{prompt}|{seed}|{steps}|{cfg}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))

    base_hue = (int(digest[16:20], 16) % 1000) / 1000.0
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    # Deterministic diagonal gradient with hue drift, evoking a denoised field.
    for y in range(height):
        for x in range(0, width, 4):
            t = (x / width + y / height) / 2.0
            hue = (base_hue + 0.25 * t) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.35 + 0.5 * t)
            color = (int(r * 255), int(g * 255), int(b * 255))
            for dx in range(4):
                if x + dx < width:
                    pixels[x + dx, y] = color

    draw = ImageDraw.Draw(image)
    # A few deterministic shapes so each prompt/seed looks distinct.
    for _ in range(6):
        x0 = rng.randint(0, width - 1)
        y0 = rng.randint(0, height - 1)
        r = rng.randint(width // 12, width // 5)
        r2, g2, b2 = colorsys.hsv_to_rgb(rng.random(), 0.7, 0.9)
        draw.ellipse(
            [x0 - r, y0 - r, x0 + r, y0 + r],
            outline=(int(r2 * 255), int(g2 * 255), int(b2 * 255)),
            width=max(2, width // 200),
        )

    prompt_hash = digest[:12]
    caption = f"[OFFLINE DEMO] {spec.repo_id}"
    prompt_line = f"prompt_id={prompt_id} prompt_hash={prompt_hash}"
    sub = f"seed={seed} steps={steps} cfg={cfg}"
    draw.rectangle([0, height - 56, width, height], fill=(0, 0, 0))
    draw.text((12, height - 50), caption, fill=(255, 255, 255))
    draw.text((12, height - 30), prompt_line, fill=(220, 220, 220))
    draw.text((12, height - 14), sub, fill=(180, 180, 180))
    return image


def _generate_kolors(
    *,
    spec: BackendSpec,
    prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    offload: bool,
    dtype_name: str | None,
):
    import torch
    from diffusers import KolorsPipeline

    _patch_kolors_tokenizer_padding_side()
    dtype = _torch_dtype(torch, dtype_name or "float16")
    pipe = KolorsPipeline.from_pretrained(
        spec.repo_id,
        torch_dtype=dtype,
        variant="fp16",
    )
    _place_pipeline(pipe, offload=offload)
    generator = torch.Generator(_generator_device(torch)).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt="",
        guidance_scale=cfg,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator,
    )
    return result.images[0]


def _patch_kolors_tokenizer_padding_side() -> None:
    """Keep Diffusers Kolors tokenizer compatible with newer Transformers.

    Transformers 4.57 passes ``padding_side`` into tokenizer ``_pad``. The
    Kolors tokenizer bundled with Diffusers 0.30.3 does not declare that
    keyword, so loading the pipeline fails before generation starts.
    """
    from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer
    from transformers.utils.generic import PaddingStrategy

    if "padding_side" in inspect.signature(ChatGLMTokenizer._pad).parameters:
        return

    original_pad = ChatGLMTokenizer._pad

    def _pad_with_padding_side(
        self,
        encoded_inputs,
        max_length=None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of=None,
        return_attention_mask=None,
        padding_side=None,
    ):
        if padding_side is None or padding_side == self.padding_side:
            return original_pad(
                self,
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        previous_padding_side = self.padding_side
        self.padding_side = padding_side
        try:
            return original_pad(
                self,
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
        finally:
            self.padding_side = previous_padding_side

    ChatGLMTokenizer._pad = _pad_with_padding_side


def _generate_lens(
    *,
    spec: BackendSpec,
    prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    aspect_ratio: str,
    offload: bool,
    dtype_name: str | None,
):
    import torch

    LensPipeline = _load_lens_pipeline_class()

    dtype = _torch_dtype(torch, dtype_name or "bfloat16")
    pipe = LensPipeline.from_pretrained(spec.repo_id, torch_dtype=dtype)
    _place_pipeline(pipe, offload=offload)
    generator = torch.Generator(_generator_device(torch)).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        base_resolution=1024,
        aspect_ratio=aspect_ratio,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    return result.images[0]


def _load_lens_pipeline_class():
    repo_path = Path(__file__).resolve().parents[2] / "external" / "Lens"
    if repo_path.exists():
        repo_path_text = str(repo_path)
        if repo_path_text not in sys.path:
            sys.path.insert(0, repo_path_text)
    try:
        from lens import LensPipeline
    except ImportError as exc:
        raise ImportError(
            "LensPipeline is unavailable. Clone https://github.com/microsoft/Lens "
            "into external/Lens or add the Lens repo to PYTHONPATH."
        ) from exc
    return LensPipeline


def _place_pipeline(pipe, *, offload: bool) -> None:
    if offload:
        pipe.enable_model_cpu_offload()
        return

    import torch

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def _generator_device(torch_module) -> str:
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _torch_dtype(torch_module, dtype_name: str):
    normalized = dtype_name.lower()
    if normalized in ("fp16", "float16"):
        return torch_module.float16
    if normalized in ("bf16", "bfloat16"):
        return torch_module.bfloat16
    if normalized in ("fp32", "float32"):
        return torch_module.float32
    raise ValueError("dtype must be float16, bfloat16, or float32.")
