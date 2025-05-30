# Ensure script is running as administrator
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Host "This script must be run as Administrator." -ForegroundColor Red
  Pause
  exit 1
}


function Write-Yellow($Text) {
  Write-Host $Text -ForegroundColor Yellow
}
function Write-Cyan($Text) {
  Write-Host $Text -ForegroundColor Cyan
}

function Pause {
  Read-Host -Prompt "Press Enter to continue"
}

# >>> Added Check <<<
Write-Host ""
Write-Yellow "[0] Preflight checks..."

# Check Python version
try {
  $pythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if (-not $pythonVersion) { throw "Python not found" }
  $versionParts = $pythonVersion.Split('.')
  if ([int]$versionParts[0] -lt 3 -or ([int]$versionParts[0] -eq 3 -and [int]$versionParts[1] -lt 12)) {
    Write-Host "   [FAILED] Python 3.12+ required. Found: $pythonVersion" -ForegroundColor Red
    exit 1
  }
  Write-Host "   -> Python version $pythonVersion found."
} catch {
  Write-Host "   [FAILED] Python not found or inaccessible in PATH." -ForegroundColor Red
  exit 1
}

# Check for CUDA and nvcc
try {
  $nvccVersion = & nvcc --version 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "   [FAILED] CUDA 'nvcc' not found in PATH. Make sure CUDA toolkit is installed." -ForegroundColor Red
    exit 1
  }
  Write-Host "   -> CUDA detected: nvcc found."
} catch {
  Write-Host "   [FAILED] CUDA 'nvcc' not found." -ForegroundColor Red
  exit 1
}

# Check for MSVC (cl.exe)
try {
  $msvc = & cl 2>&1 | Out-String
  if ($msvc -match "Microsoft.*C\+\+") {
    Write-Host "   -> MSVC detected: cl.exe available."
  } else {
    throw "MSVC not found"
  }
} catch {
  Write-Host "   [FAILED] MSVC compiler (cl.exe) not found in PATH. Install Build Tools for Visual Studio." -ForegroundColor Red
  exit 1
}
Pause

Write-Host ""
Write-Yellow "[1] Creating Python virtual environment '.venv'..."
if (-Not (Test-Path .venv)) {
  python -m venv .venv
  Write-Host "   -> Virtual environment created."
} else {
  Write-Host "   -> .venv already exists, skipping."
}

Write-Host ""
Write-Yellow "[2] Activating venv and configuring environment..."
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
Write-Yellow "[3] Installing Poetry into the venv..."
# If you installed Python via the Microsoft Store, use 'python' instead of 'py' below.
$pythonCmd = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | & $pythonCmd -
# Find Poetry executable path
$poetryPath = $null
$possiblePaths = @(
  "$env:POETRY_HOME\bin\poetry",
  "$env:POETRY_HOME\venv\Scripts\poetry",
  "$PWD\.venv\bin"
  "$env:APPDATA\Python\Scripts\poetry",
  "$env:APPDATA\pypoetry\venv\Scripts\poetry"
)
foreach ($path in $possiblePaths) {
  if ($env:POETRY_HOME -and (Test-Path $path)) {
    $poetryPath = $path
    break
  }
}
if (-not $poetryPath) {
  $userProfile = $env:USERPROFILE
  $morePaths = @(
    "$userProfile\.local\bin\poetry",
    "$userProfile\.local\share\pypoetry\venv\bin\poetry",
    "$userProfile\Library\Application Support\pypoetry\venv\bin\poetry"
  )
  foreach ($path in $morePaths) {
    if (Test-Path $path) {
      $poetryPath = $path
      break
    }
  }
}
if (-not $poetryPath) {
  Write-Host "   [FAILED] Poetry not found. Ensure it was installed correctly and added to path." -ForegroundColor Red
  Write-Host "   See the official Poetry installation guide for help: https://python-poetry.org/docs/#installing-with-the-official-installer" -ForegroundColor Yellow
  exit 1
}
Write-Host "   -> Poetry installed. ($poetryPath)"
Pause

Write-Host ""
Write-Yellow "[4] Installing Python dependencies via Poetry..."
poetry install
Write-Host "   -> All Python packages installed."
Pause

Write-Host ""
Write-Yellow "[5] Building native backend (pawX)..."
Write-Host "   -> Changing directory to pawX"
Push-Location pawX
.\build.ps1
Pop-Location
Write-Host "   -> pawX build complete. Check pawX\ for the .pyd file."
Write-Host ""
Write-Cyan "[OK] Setup finished! You can now run 'poetry run python your_script.py'"
