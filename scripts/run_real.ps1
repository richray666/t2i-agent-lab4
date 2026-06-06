# Real generation with pretrained models on a GPU machine.
# Keeps the Hugging Face cache inside this lab folder (hf_cache/).
# First Kolors run downloads ~17GB; allow a long window.
#
# Run from the project root:
#     ./scripts/run_real.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# Kolors/SentencePiece can fail on non-ASCII paths. Use an ASCII subst drive
# when the lab folder path contains Chinese characters.
$runRoot = $root
if ($root -match '[^\x00-\x7F]') {
    $subst = cmd /c subst
    $expected = "X:\: => $root"
    if ($subst -match '^X:\\: => ') {
        if ($subst -notcontains $expected) {
            throw "X: is already mapped to another path. Run manually from an ASCII path or choose another subst drive."
        }
    } else {
        cmd /c subst X: "$root"
    }
    $runRoot = "X:\"
}

Set-Location $runRoot
$env:PYTHONUTF8 = "1"
$env:HF_HOME = (Join-Path $runRoot 'hf_cache')
$env:PYTHONPATH = (Join-Path $runRoot 'src')
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PATH = "C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\Library\bin;C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\bin;" + $env:PATH
$python = "C:\Users\lenovo\anaconda3\envs\pytorch2.2.2\python.exe"

Write-Host "=== Kolors (Chinese, cn_scene smoke test) ==="
& $python -X utf8 -m t2i_agent generate `
    --backend kolors --prompt-id cn_scene --seed 66 `
    --steps 4 --width 768 --height 768 --offload --dtype float16

Write-Host "`nKolors smoke test finished. See the outputs/ folder."
Write-Host "Lens-Turbo is optional because it downloads about 29GB. Run manually if needed:"
Write-Host "& $python -X utf8 -m t2i_agent generate --backend lens --variant turbo --prompt-id en_scene --seed 42 --offload"
