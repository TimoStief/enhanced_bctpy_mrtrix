# Launch the BCT Analysis Web Interface
# Using UV for environment management

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "BCT Analysis Web Interface (UV)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if UV is installed
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "[INFO] Using uv to run application..." -ForegroundColor Cyan
    Set-Location $PROJECT_ROOT
    
    # Sync dependencies and run the app
    # Use --project to ensure it finds pyproject.toml
    uv run --project . python web_app/app.py
}
else {
    Write-Host "[WARNING] uv not found in PATH. Falling back to manual venv..." -ForegroundColor Yellow
    
    $VIRTUAL_ENV = Join-Path $PROJECT_ROOT ".venv"
    $PYTHON = Join-Path $VIRTUAL_ENV "Scripts\python.exe"

    if (-not (Test-Path $PYTHON)) {
        Write-Host "[ERROR] Virtual environment not found and uv is missing." -ForegroundColor Red
        Write-Host "[INFO] Please run: python scripts/setup_env.py" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "[SUCCESS] Starting web interface..." -ForegroundColor Green
    Set-Location (Join-Path $PROJECT_ROOT "web_app")
    & $PYTHON app.py
}
