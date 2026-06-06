# 实验四：文生图智能体

本项目实现了“基于预训练模型的文生图实践-实验4”要求的小型文生图智能体。智能体可以统一调用国产开源模型 Kolors 和国外开源模型 Microsoft Lens / Lens-Turbo，完成提示词读取、后端选择、图片生成、元数据保存和错误记录。

## 项目结构

- `src/t2i_agent/`：智能体源码，包括提示词加载、后端路由、模型调用、图片保存和元数据记录。
- `prompts/test_prompts.jsonl`：5 条测试提示词，覆盖中文场景、英文场景、多属性绑定、文字渲染和风格化生成。
- `scripts/`：运行脚本，包括批量运行脚本 `run_all.py`、离线演示脚本 `run_demo.ps1` 和真实模型脚本 `run_real.ps1`。
- `outputs/`：生成图片、同名 JSON 元数据和错误记录。
- `reports/实验四_基于预训练模型的文生图实践报告.md`：实验报告。
- `RUNBOOK.md`：运行手册，说明测试、离线演示和真实模型生成步骤。
- `environment.yml`：推荐的 conda 环境配置。

## 支持的后端

- `offline`：本地离线演示后端，只依赖 PIL，不下载模型、不需要 GPU。它会生成真实的 png+json 文件，用于验证整条流程，但图像质量不代表真实模型。
- `kolors`：快手 Kolors，模型仓库为 `Kwai-Kolors/Kolors-diffusers`，权重约 17GB，适合中文文生图。
- `lens`：微软 Lens / Lens-Turbo，模型仓库为 `microsoft/Lens` 或 `microsoft/Lens-Turbo`，权重约 29GB，适合英文文生图。

> 注意：本仓库不包含 Microsoft Lens 源码和大模型权重。Lens 源码需要按需克隆到 `external/Lens`：
>
> ```bash
> git clone https://github.com/microsoft/Lens external/Lens
> ```
>
> 模型权重会按需下载到 `hf_cache/`，该目录已被 `.gitignore` 忽略。Lens-Turbo 的文本编码器体量较大，本实验中的 Lens-Turbo 图片是在远程高显存 GPU 服务器上生成的；本机 8GB 显存更适合运行 Kolors 单图冒烟测试。

## 快速开始：离线演示

离线演示不需要下载模型，适合先验证代码流程：

```powershell
python scripts/run_all.py --backend offline --seed 66
```

也可以只生成一张图：

```powershell
python -m t2i_agent generate --backend offline --prompt-id cn_scene --seed 66
```

生成结果会保存到 `outputs/`，每张图片都有同名 `.json` 元数据。

## 环境配置

推荐使用 conda 创建实验环境：

```powershell
conda env create -f environment.yml
conda activate t2i-lab4
```

当前本机也可以使用已有的 `pytorch2.2.2` 环境运行测试和 Kolors 冒烟：

```powershell
conda run -n pytorch2.2.2 python -m pip install -e .
```

## 单元测试

```powershell
conda run -n pytorch2.2.2 python -m unittest discover -s tests -v
```

测试内容包括提示词字段校验、后端参数解析、元数据保存、错误记录、离线后端生成和确定性检查。

## 真实模型生成

真实模型建议使用本项目目录内的 Hugging Face 缓存，避免权重下载到系统其他位置：

```powershell
cmd /c subst X: "D:\数字媒体处理技术实验三"
Set-Location X:\
$env:PYTHONUTF8="1"
$env:HF_HOME="X:\hf_cache"
$env:PYTHONPATH="X:\src"
$env:PATH="C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\Library\bin;C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\bin;" + $env:PATH
$python="C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\python.exe"
```

Kolors 中文单图生成：

```powershell
& $python -X utf8 -m t2i_agent generate --backend kolors --prompt-id cn_scene --seed 66 --steps 4 --width 768 --height 768 --offload --dtype float16
```

Lens-Turbo 英文单图生成：

```powershell
& $python -X utf8 -m t2i_agent generate --backend lens --variant turbo --prompt-id en_scene --seed 42 --offload
```

默认真实模型冒烟建议先跑 Kolors 单图。Lens-Turbo 首次运行需要下载约 29GB 权重，本机 8GB 显存风险较高；如果显存不足，应开启 `--offload`、降低分辨率，并一次只跑一张图。

## 已生成结果

本项目已保留实验生成结果：

- Kolors：`outputs/20260605-005052-kolors-default-cn_scene-seed66.png`
- Lens-Turbo：`outputs/20260605-115614-lens-turbo-en_scene-seed42.png` 及其他测试提示词结果
- Offline：5 张离线流程验证图

每张图片都有对应 JSON 元数据，记录模型、提示词、随机种子、采样步数、CFG、分辨率、运行时间和 offload 状态。
