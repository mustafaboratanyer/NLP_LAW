$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python bulunamadı. Python 3.11 veya 3.12 kurulduktan sonra tekrar çalıştırın."
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama bulunamadı. https://ollama.com/download adresinden Ollama kurulmalıdır."
}

$publicRoot = if ($env:PUBLIC) { $env:PUBLIC } else { $env:ProgramData }
$ollamaModels = Join-Path $publicRoot "NLP_LAW\ollama_models"
New-Item -ItemType Directory -Force -Path $ollamaModels | Out-Null
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ollamaModels, "User")
$env:OLLAMA_MODELS = $ollamaModels

Get-Process | Where-Object {
    $_.ProcessName -in @("ollama", "ollama app")
} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$ollamaExecutable = (Get-Command ollama).Source
Start-Process -FilePath $ollamaExecutable -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 4

Write-Host "Python bağımlılıkları kuruluyor..."
python -m pip install -r requirements.txt

Write-Host "Qwen2.5 7B base modeli indiriliyor..."
ollama pull qwen2.5:7b-instruct-q4_K_M

$adapterPath = Join-Path $repoRoot "ollama\qwen2.5-7b-law-lora-f16.gguf"
if (-not (Test-Path -LiteralPath $adapterPath)) {
    throw "Fine-tuned GGUF adapter bulunamadı: $adapterPath"
}

Write-Host "Fine-tuned Ollama modeli oluşturuluyor..."
ollama create nlp-law-finetuned -f ollama\Modelfile.finetuned

Write-Host ""
Write-Host "Kurulum tamamlandı."
Write-Host "Arayüzü başlatmak için: .\start_demo.ps1"
