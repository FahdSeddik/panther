function Write-Blue($Text) {
  Write-Host $Text -ForegroundColor Blue
}
function Write-Green($Text) {
  Write-Host $Text -ForegroundColor Green
}

param()

function Pause {
  Read-Host -Prompt "Press Enter to continue"
}

Write-Host ""
Write-Blue "[1] Creating Python virtual environment '.venv'..."
if (-Not (Test-Path .venv)) {
  python -m venv .venv
  Write-Host "   -> Virtual environment created."
} else {
  Write-Host "   -> .venv already exists, skipping."
}

Write-Host ""
Write-Blue "[2] Activating venv and configuring environment..."
# Activate the venv
& ".\.venv\Scripts\Activate.ps1"
# Ensure Poetry installs into the venv
$env:POETRY_HOME = "$PWD\.venv"
# Prepend venv Scripts and Poetry bin to PATH
$poetryBin = Join-Path $env:POETRY_HOME 'venv\Scripts'
$env:Path = "$poetryBin;$env:Path"
Write-Host "   -> venv activated; POETRY_HOME set to $env:POETRY_HOME"
Write-Host "   -> Updated PATH to include venv Scripts and Poetry bin"
Pause

Write-Host ""
Write-Blue "[3] Installing Poetry into the venv..."
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content |
  py - --verbose
Write-Host "   -> Poetry installed. ($(Get-Command poetry).Source)"
Pause

Write-Host ""
Write-Blue "[4] Installing Python dependencies via Poetry..."
poetry install
Write-Host "   -> All Python packages installed."
Pause

Write-Host ""
Write-Blue "[5] Building native backend (pawX)..."
Write-Host "   -> Changing directory to pawX"
Push-Location pawX
.\build.ps1
Pop-Location
Write-Host "   -> pawX build complete. Check pawX\ for the .pyd file."
Write-Host ""
Write-Green "[OK] Setup finished! You can now run 'poetry run python your_script.py'"
