# install.ps1
# Bootstrap Python environment with proper EXE installation

# -------------------------------
# Elevate if not admin
# -------------------------------
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "Relaunching with administrator privileges..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Install-Python {
    Write-Host "Checking for Python installation..."

    # Detect existing
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonPath = (Get-Command python).Source
        if ($pythonPath -and -not ($pythonPath -like "*WindowsApps*")) {
            return "python"
        }
    }

    Write-Host "Python not found. Installing from official EXE..."

    $exeUrl = "https://www.python.org/ftp/python/3.13.7/python-3.13.7-amd64.exe"
    $installerPath = "$env:TEMP\python-3.13.7-amd64.exe"

    # Download EXE
    if (-not (Test-Path $installerPath) -or ((Get-Item $installerPath).Length -lt 20000000)) {
        Write-Host "Downloading Python installer..."
        Invoke-WebRequest -Uri $exeUrl -OutFile $installerPath -UseBasicParsing
    }

    if (-not (Test-Path $installerPath)) {
        Write-Error "Download failed. Could not find $installerPath"
        exit 1
    }

    Write-Host "Installing Python silently (this may take up to 2 minutes)..."
    $args = "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0"
    $proc = Start-Process -FilePath $installerPath -ArgumentList $args -Wait -PassThru

    # Check exit code
    if ($proc.ExitCode -ne 0) {
        Write-Host "Installer returned exit code $($proc.ExitCode). Trying to reinstall with GUI..."
        Start-Process -FilePath $installerPath -ArgumentList "InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0" -Wait
    }

    # Verify installation
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and -not ($pythonCmd.Source -like "*WindowsApps*")) {
        Write-Host "Python installed successfully at $($pythonCmd.Source)"
        return "python"
    }

    Write-Error "Python installation failed."
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

    $pythonCmd = Install-Python

    $venvPath = Join-Path $scriptDir 'venv'
    if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
        Write-Host "Creating virtual environment..."
        & $pythonCmd -m venv $venvPath
    }

    $venvPython = Join-Path $venvPath 'Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        Write-Error "Virtual environment not created correctly."
        exit 1
    }

    Write-Host "Upgrading pip/setuptools/wheel..."
    & $venvPython -m pip install --upgrade pip setuptools wheel --disable-pip-version-check

    Write-Host "Installing dependencies..."
    & $venvPython -m pip install -r $requirementsFile --disable-pip-version-check

    Write-Host "`n✅ Done. Activate the environment with:`n`t .\venv\Scripts\Activate.ps1"
}
finally {
    Pop-Location
}
