# install.ps1
# Bootstrap Python virtual environment and install dependencies from requirements.txt
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve script dir and files
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir
try {
    $requirementsFile = Join-Path $scriptDir 'requirements.txt'
    if (-not (Test-Path $requirementsFile)) {
        Write-Error "requirements.txt not found next to install.ps1"
        exit 1
    }

    # Locate Python (prefer py launcher on Windows)
    $pythonCmd = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCmd = 'py'
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCmd = 'python'
    } else {
        Write-Error "Python is not installed or not on PATH. Please install Python 3.10+ from https://www.python.org/downloads/ and re-run."
        exit 1
    }

    # Create venv if missing
    $venvPath = Join-Path $scriptDir 'venv'
    if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
        Write-Host "Creating virtual environment at $venvPath ..."
        & $pythonCmd -3 -m venv $venvPath 2>$null
        if ($LASTEXITCODE -ne 0) {
            # Fallback without -3 switch
            & $pythonCmd -m venv $venvPath
        }
    }

    # Build pip command inside venv
    $venvPython = Join-Path $venvPath 'Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        Write-Error "Virtual environment not created correctly."
        exit 1
    }
    $pipCmd = "$venvPython -m pip"

    # Upgrade pip/setuptools/wheel
    Write-Host "Upgrading pip/setuptools/wheel ..."
    & $venvPython -m pip install --upgrade pip setuptools wheel --disable-pip-version-check

    # Install all requirements in one shot (faster and resolves deps better)
    Write-Host "Installing dependencies from requirements.txt ..."
    & $venvPython -m pip install -r $requirementsFile --disable-pip-version-check

    Write-Host "Done. To activate the venv for this session, run:`n`t .\\venv\\Scripts\\Activate.ps1"
}
finally {
    Pop-Location
}