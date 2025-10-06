# install.ps1
# Bootstrap Python virtual environment and install dependencies from requirements.txt

# -------------------------------
# Elevate script if not running as admin
# -------------------------------
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "Script not running as administrator. Relaunching with elevated privileges..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -------------------------------
# Helper: install Python if missing or only Microsoft Store alias
# -------------------------------
function Install-Python {
    Write-Host "Checking for Python installation..."

    # Detect if py launcher exists
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }

    # Detect if python.exe exists and is real (no alias to Microsoft Store)
    $pythonPath = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
    if ($pythonPath -and -not ($pythonPath -like "*WindowsApps*")) {
        return "python"
    }

    Write-Host "Python not found or only Microsoft Store alias detected. Installing real Python..."

    # Try winget first
    try {
        winget install --id Python.Python.310 -e --silent
    }
    catch {
        Write-Host "winget failed or unavailable. Trying Chocolatey..."

        # Ensure Chocolatey installed
        if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
            Write-Host "Installing Chocolatey..."
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        }

        choco install python --version=3.10 -y
    }

    # Verify Python now exists
    $pythonPath = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
    if ($pythonPath -and -not ($pythonPath -like "*WindowsApps*")) {
        return "python"
    }

    Write-Error "Python installation failed. Please install manually from https://www.python.org/downloads/"
    exit 1
}

# -------------------------------
# Main
# -------------------------------
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    $requirementsFile = Join-Path $scriptDir 'requirements.txt'
    if (-not (Test-Path $requirementsFile)) {
        Write-Error "requirements.txt not found next to install.ps1"
        exit 1
    }

    # Ensure Python installed
    $pythonCmd = Install-Python

    # Create venv if missing
    $venvPath = Join-Path $scriptDir 'venv'
    if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
        Write-Host "Creating virtual environment at $venvPath ..."
        & $pythonCmd -m venv $venvPath
    }

    # Verify venv
    $venvPython = Join-Path $venvPath 'Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        Write-Error "Virtual environment not created correctly."
        exit 1
    }

    # Upgrade pip/setuptools/wheel
    Write-Host "Upgrading pip/setuptools/wheel ..."
    & $venvPython -m pip install --upgrade pip setuptools wheel --disable-pip-version-check

    # Install dependencies
    Write-Host "Installing dependencies from requirements.txt ..."
    & $venvPython -m pip install -r $requirementsFile --disable-pip-version-check

    Write-Host "Done. To activate the venv for this session, run:`n`t .\\venv\\Scripts\\Activate.ps1"
}
finally {
    Pop-Location
}
