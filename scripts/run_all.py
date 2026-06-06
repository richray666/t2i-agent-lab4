"""Batch-run every test prompt through one backend.

Examples
--------
Offline demo (no model download, works anywhere):
    python scripts/run_all.py --backend offline --seed 66

Real Kolors generation (needs GPU + downloaded weights):
    python scripts/run_all.py --backend kolors --seed 66 --offload --steps 25

The script never aborts the whole batch on a single failure: each failed
prompt is recorded as an error JSON (via the agent's own error handling) and
the run continues to the next prompt.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from t2i_agent.generator import GenerateOptions, generate  # noqa: E402
from t2i_agent.metadata import save_error_record  # noqa: E402
from t2i_agent.prompts import load_prompts  # noqa: E402

PROMPT_FILE = PROJECT_ROOT / "prompts" / "test_prompts.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all prompts through one backend.")
    parser.add_argument("--backend", choices=["offline", "kolors", "lens"], default="offline")
    parser.add_argument("--variant", choices=["default", "demo", "turbo", "full"])
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--language", choices=["zh", "en"])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args(argv)

    prompts = load_prompts(PROMPT_FILE)
    ok, failed = 0, 0
    for prompt_id in prompts:
        print(f"\n=== {args.backend} :: {prompt_id} ===")
        options = GenerateOptions(
            backend=args.backend,
            variant=None if args.variant == "default" else args.variant,
            prompt_id=prompt_id,
            prompt_file=PROMPT_FILE,
            output_dir=OUTPUT_DIR,
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            width=args.width,
            height=args.height,
            language=args.language,
            offload=args.offload,
            dtype=args.dtype,
        )
        try:
            image_path, metadata_path = generate(options)
            print(f"  image:    {image_path}")
            print(f"  metadata: {metadata_path}")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - record and continue
            error_path = save_error_record(
                OUTPUT_DIR,
                backend=args.backend,
                prompt_id=prompt_id,
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                options=vars(args),
            )
            print(f"  FAILED -> {error_path}")
            failed += 1

    print(f"\nDone. {ok} succeeded, {failed} failed. Output dir: {OUTPUT_DIR}")
    return 1 if failed and ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
