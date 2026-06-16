$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama bulunamadı. Önce .\setup_demo.ps1 çalıştırılmalıdır."
}

$publicRoot = if ($env:PUBLIC) { $env:PUBLIC } else { $env:ProgramData }
$env:OLLAMA_MODELS = Join-Path $publicRoot "NLP_LAW\ollama_models"

if (-not (Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath (Get-Command ollama).Source -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

$installedModels = ollama list
if ($installedModels -notmatch "qwen2.5:7b-instruct-q4_K_M") {
    throw "Base Ollama modeli eksik. Önce .\setup_demo.ps1 çalıştırılmalıdır."
}
if ($installedModels -notmatch "nlp-law-finetuned") {
    throw "Fine-tuned Ollama modeli eksik. Önce .\setup_demo.ps1 çalıştırılmalıdır."
}

python -m streamlit run app.py
