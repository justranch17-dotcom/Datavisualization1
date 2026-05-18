$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Join-Path $ProjectRoot "streamlit_codex_phone_starter\streamlit_codex_phone_starter"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment not found at $Python"
}

Write-Host "Starting phone control app..."
Write-Host "Project: $ProjectRoot"
Write-Host "Local URL: http://localhost:8501"
Write-Host "Tailscale URL will be: http://YOUR-TAILSCALE-NAME:8501"

Push-Location $AppDir
try {
    & $Python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
}
finally {
    Pop-Location
}
