# install.ps1
# This script installs the Python packages listed in requirements.txt

# Get the directory of the script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Construct the full path to requirements.txt
$requirementsFile = Join-Path $scriptDir "requirements.txt"

# Check if requirements.txt exists
if (-not (Test-Path $requirementsFile)) {
    Write-Error "requirements.txt not found in the script's directory."
    exit 1
}

# Read the requirements file and install each package
Get-Content $requirementsFile | ForEach-Object {
    $package = $_.Trim()
    if ($package) {
        Write-Host "Installing $package..."
        pip install $package
    }
}

Write-Host "All packages installed."