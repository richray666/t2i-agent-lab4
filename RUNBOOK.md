# 运行手册 (RUNBOOK)

本文件说明如何在本机运行实验四的文生图智能体，包含三种用途：跑单元测试、离线演示、真实模型生成。

## 0. 三种后端

| backend | 说明 | 是否需要下载/GPU |
| --- | --- | --- |
| `offline` | 本地确定性占位图后端，用 PIL 直接合成图片。用于验证整条流水线、出真实 png+json 产物。 | 否 |
| `kolors` | 快手 Kolors（约 17GB），中文文生图。 | 需要，建议 GPU |
| `lens` | 微软 Lens / Lens-Turbo（约 29GB），英文文生图。 | 需要，建议 GPU |

## 1. 单元测试

```powershell
conda run -n pytorch2.2.2 python -m unittest discover -s tests -v
```

测试覆盖：提示词加载与字段校验、后端默认参数、产物保存、离线后端端到端生成与可复现性。

## 2. 离线演示（任何机器都能跑，立刻出图）

```powershell
# 方式一：脚本
./scripts/run_demo.ps1

# 方式二：单条命令
python scripts/run_all.py --backend offline --seed 66

# 方式三：只生成一张
python -m t2i_agent generate --backend offline --prompt-id cn_scene --seed 66
```

产物写入 `outputs/`，每张图都有同名 `.json` 元数据。离线图带 `[OFFLINE DEMO]` 水印，相同 (prompt, seed) 结果完全一致，用于证明流水线可用，但**不是**真实模型质量，报告里需注明。

## 3. 真实模型生成（GPU 机器）

```powershell
# 默认只跑 Kolors 单图冒烟，避免一次性触发 Lens-Turbo 的 29GB 下载
./scripts/run_real.ps1
```

或手动逐步：

```powershell
# 如果项目目录含中文路径，先创建 ASCII 盘符映射，避免 SentencePiece 路径乱码。
# 已存在相同映射时可跳过；不要删除任何缓存。
cmd /c subst X: "D:\数字媒体处理技术实验三"
Set-Location X:\

$env:PYTHONUTF8 = "1"
$env:HF_HOME = "X:\hf_cache"   # 缓存留在本目录，可断点续传
$env:PYTHONPATH = "X:\src"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PATH = "C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\Library\bin;C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\bin;" + $env:PATH
$python = "C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\python.exe"

# Kolors 中文单图冒烟
& $python -X utf8 -m t2i_agent generate `
    --backend kolors --prompt-id cn_scene --seed 66 `
    --steps 4 --width 768 --height 768 --offload --dtype float16

# Lens-Turbo 英文，可选。首次运行约 29GB，建议 Kolors 成功后再单独决定是否跑。
& $python -X utf8 -m t2i_agent generate `
    --backend lens --variant turbo --prompt-id en_scene --seed 42 --offload
```

显存不足时：加 `--offload`、降到 `--width 768 --height 768`、一次只跑一张。
首次运行会下载 ~17GB / ~29GB，请预留磁盘和时间；`hf_cache/` 已保留部分缓存，会续传。

## 4. 失败如何记录

任何一步失败（下载超时、HF 权限、显存不足等）都会在 `outputs/errors/` 写一条错误 JSON，而不是静默忽略。这些记录本身也是实验现象，可写进报告。
