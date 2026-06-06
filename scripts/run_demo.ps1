# Offline demo: generates real png+json for all 5 prompts, no model download.
# Run from the project root in PowerShell:
#     ./scripts/run_demo.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
python scripts/run_all.py --backend offline --seed 66
Write-Host "`nOffline demo finished. See the outputs/ folder."
