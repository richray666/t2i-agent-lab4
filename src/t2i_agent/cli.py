from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from .generator import GenerateOptions, generate
from .metadata import save_error_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "test_prompts.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m t2i_agent",
        description="Experiment 4 text-to-image agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate one image.")
    generate_parser.add_argument(
        "--backend", choices=["offline", "kolors", "lens"], required=True
    )
    generate_parser.add_argument("--variant", choices=["default", "demo", "turbo", "full"])
    generate_parser.add_argument("--prompt-id", required=True)
    generate_parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    generate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--steps", type=int)
    generate_parser.add_argument("--cfg", type=float)
    generate_parser.add_argument("--width", type=int)
    generate_parser.add_argument("--height", type=int)
    generate_parser.add_argument("--aspect-ratio", default=None)
    generate_parser.add_argument("--language", choices=["zh", "en"])
    generate_parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    generate_parser.add_argument("--offload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        options = GenerateOptions(
            backend=args.backend,
            variant=None if args.variant == "default" else args.variant,
            prompt_id=args.prompt_id,
            prompt_file=args.prompt_file,
            output_dir=args.output_dir,
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            width=args.width,
            height=args.height,
            aspect_ratio=args.aspect_ratio,
            language=args.language,
            offload=args.offload,
            dtype=args.dtype,
        )
        try:
            image_path, metadata_path = generate(options)
        except Exception as exc:
            error_path = save_error_record(
                args.output_dir,
                backend=args.backend,
                prompt_id=args.prompt_id,
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                options=vars(args),
            )
            print(f"Generation failed. Error record: {error_path}")
            raise
        print(f"Image saved: {image_path}")
        print(f"Metadata saved: {metadata_path}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
