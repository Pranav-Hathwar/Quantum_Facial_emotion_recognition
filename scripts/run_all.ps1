# QuantumVision - one-shot setup + training on Windows (NVIDIA GPU). Run from this folder in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass ; .\run_all.ps1 -Zip C:\path\to\archive.zip
param([Parameter(Mandatory=$true)][string]$Zip, [int[]]$Seeds = @(42,43,44))
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
. .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -m ml.preprocessing.zip_to_folders --zip $Zip
python scripts/fetch_face_model.py
python extract_features.py --device cuda
python run_experiments.py --seeds $Seeds --device cuda
python compare.py
Write-Host "Done. Start the app: uvicorn backend.app.main:app --port 8000  (then: cd frontend; npm install; npm run dev)"
