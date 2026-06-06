# Experiment 4 Text-to-Image Agent

This project implements the lab-4 agent described in `基于预训练模型的文生图实践-实验4.pptx`.

## Files

- `src/t2i_agent/`: Python package for prompt loading, backend routing, generation, and metadata records.
- `prompts/test_prompts.jsonl`: five test prompts for Chinese scene understanding, English scene generation, attribute binding, text rendering, and stylized generation.
- `scripts/`: batch runner (`run_all.py`) and PowerShell helpers (`run_demo.ps1` offline, `run_real.ps1` GPU models).
- `outputs/`: generated images, metadata JSON files, and error records.
- `reports/实验四_基于预训练模型的文生图实践报告.md`: experiment report.
- `RUNBOOK.md`: step-by-step run guide (tests / offline demo / real models).
- `environment.yml`: recommended conda environment.

## Backends

- `offline`: deterministic placeholder backend (PIL only, no download/GPU). Runs the full pipeline end-to-end and writes real png+json, for verifying the agent anywhere. Image quality is not representative of a real model.
- `kolors`: Kwai Kolors (~17GB), Chinese text-to-image.
- `lens`: Microsoft Lens / Lens-Turbo (~29GB), English text-to-image.

> **Note on the `lens` backend:** the Microsoft Lens source code and the model
> weights are **not** included in this repository (see `.gitignore`). To use the
> `lens` backend, clone the upstream repo into `external/Lens` first:
>
> ```bash
> git clone https://github.com/microsoft/Lens external/Lens
> ```
>
> Model weights (`Kwai-Kolors/Kolors-diffusers`, `microsoft/Lens-Turbo`) are
> downloaded on demand into `hf_cache/` (also git-ignored). Lens-Turbo's text
> encoder is large, so it was run on a remote multi-GPU server rather than an
> 8 GB laptop (see `reports/`).

## Quick start (offline demo, no download)

```powershell
python scripts/run_all.py --backend offline --seed 66
# or a single image:
python -m t2i_agent generate --backend offline --prompt-id cn_scene --seed 66
```

## Environment

Recommended fresh environment:

```powershell
conda env create -f environment.yml
conda activate t2i-lab4
```

The current machine also has a working `pytorch2.2.2` conda environment with CUDA-enabled PyTorch, Diffusers, Transformers, Accelerate, Safetensors, and Pillow. It can run tests and may run the agent after editable install:

```powershell
conda run -n pytorch2.2.2 python -m pip install -e .
```

## Unit Tests

```powershell
conda run -n pytorch2.2.2 python -m unittest discover -s tests -v
```

## Generation Commands

Use a workspace-local Hugging Face cache so downloads stay inside this lab folder:

```powershell
cmd /c subst X: "D:\数字媒体处理技术实验三"
Set-Location X:\
$env:PYTHONUTF8="1"
$env:HF_HOME="X:\hf_cache"
$env:PYTHONPATH="X:\src"
$env:PATH="C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\Library\bin;C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\bin;" + $env:PATH
$python="C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\python.exe"
```

Kolors Chinese text-to-image:

```powershell
& $python -X utf8 -m t2i_agent generate --backend kolors --prompt-id cn_scene --seed 66 --steps 4 --width 768 --height 768 --offload --dtype float16
```

Lens-Turbo English text-to-image:

```powershell
& $python -X utf8 -m t2i_agent generate --backend lens --variant turbo --prompt-id en_scene --seed 42 --offload
```

The default real-model smoke path is Kolors single-image generation. Lens-Turbo is optional because it downloads about 29 GB. If VRAM is insufficient, retry with `--offload`, lower resolution, and one prompt at a time. Lens may require Hugging Face access to gated dependencies.

Model-size note from Hugging Face metadata:

- `Kwai-Kolors/Kolors-diffusers`: about 17 GB.
- `microsoft/Lens-Turbo`: about 29 GB.

Kolors smoke output from this workspace: `outputs/20260605-005052-kolors-default-cn_scene-seed66.png` with metadata `outputs/20260605-005052-kolors-default-cn_scene-seed66.json`.
