# start.ps1
# This script activates the virtual environment and runs the main Python application.

# Get the directory of the script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Construct the path to the virtual environment's activation script
$activateScript = Join-Path $scriptDir "venv\Scripts\Activate.ps1"

# Check if the activation script exists
if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment activation script not found at '$activateScript'."
    Write-Error "Please make sure the virtual environment exists in the 'venv' directory."
    exit 1
}

# Activate the virtual environment
. $activateScript

Write-Host "Virtual environment activated."

# Construct the path to the main Python script
$mainScript = Join-Path $scriptDir "hacktinver.py"

# Check if the main script exists
if (-not (Test-Path $mainScript)) {
    Write-Error "Main Python script 'hacktinver.py' not found."
    exit 1
}

# Run the main Python script
Write-Host "Running hacktinver.py..."
python $mainScript

Write-Host "Script finished."