# install.ps1
# Robust installer: Python (EXE -> winget -> choco), venv, pip deps with prefer-binary,
# installs MS Build Tools if necessary and retries.

# -------------------------------
# Relaunch elevated if not admin
# -------------------------------
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "No se ejecuta como administrador. Relanzando con privilegios elevados..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -------------------------------
# Helpers
# -------------------------------
function Write-Stamp($msg) {
    $t = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "[$t] $msg"
}

function Command-Exists($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Python-IsReal() {
    $c = Get-Command python -ErrorAction SilentlyContinue
    if (-not $c) { return $false }
    if ($c.Source -like "*WindowsApps*") { return $false }
    return $true
}

function Install-BuildTools {
    Write-Stamp "Instalando Visual Studio Build Tools (si no existe)..."
    if (-not (Command-Exists "vswhere.exe")) {
        # Try winget first
        try {
            Write-Stamp "Intentando instalar Build Tools con winget..."
            winget install --id Microsoft.VisualStudio.2022.BuildTools -e --silent --accept-source-agreements --accept-package-agreements
            return
        } catch {
            Write-Stamp "winget no disponible o falló al instalar Build Tools. Intentando con Chocolatey..."
        }

        # Fallback: Chocolatey
        if (-not (Command-Exists "choco")) {
            Write-Stamp "Instalando Chocolatey..."
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        }
        choco install visualstudio2019buildtools -y --no-progress
    } else {
        Write-Stamp "vswhere.exe ya presente."
    }
}

# -------------------------------
# Install Python (EXE -> winget -> choco)
# -------------------------------
function Install-Python {
    Write-Stamp "Verificando instalación de Python..."
    if (Command-Exists "py") { Write-Stamp "Se detectó launcher 'py'."; return "py" }
    if (Python-IsReal) { Write-Stamp "Python real ya instalado."; return "python" }

    # 1) Try official EXE (silent)
    $exeUrl = "https://www.python.org/ftp/python/3.13.7/python-3.13.7-amd64.exe"
    $installerPath = Join-Path $env:TEMP "python-3.13.7-amd64.exe"

    try {
        Write-Stamp "Descargando instalador oficial de Python..."
        Invoke-WebRequest -Uri $exeUrl -OutFile $installerPath -UseBasicParsing -ErrorAction Stop

        # Quick sanity: size > ~20MB
        $size = (Get-Item $installerPath).Length
        if ($size -lt 20MB) {
            Write-Stamp "Archivo descargado parece pequeño ($size bytes). Abortando intento EXE."
            Remove-Item -Force $installerPath -ErrorAction SilentlyContinue
            throw "Downloaded file too small"
        }

        Write-Stamp "Ejecutando instalador EXE en modo silencioso..."
        $args = "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0"
        $proc = Start-Process -FilePath $installerPath -ArgumentList $args -Wait -PassThru

        Write-Stamp "Instalador EXE finalizó con código de salida $($proc.ExitCode)."
        Start-Sleep -Seconds 2
        if (Python-IsReal) { return "python" }
    } catch {
        Write-Stamp "Instalación vía EXE falló o no fue concluyente: $($_.Exception.Message)"
    }

    # 2) Try winget
    try {
        Write-Stamp "Intentando instalar con winget (Python.Python.3.13)..."
        winget install --id Python.Python.3.13 -e --silent --accept-source-agreements --accept-package-agreements
        if (Python-IsReal) { return "python" }
    } catch {
        Write-Stamp "winget no disponible o falló: $($_.Exception.Message)"
    }

    # 3) Try Chocolatey
    try {
        Write-Stamp "Intentando instalar con Chocolatey..."
        if (-not (Command-Exists "choco")) {
            Write-Stamp "Instalando Chocolatey..."
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        }
        choco install python --version=3.13 -y --no-progress
        if (Python-IsReal) { return "python" }
    } catch {
        Write-Stamp "Chocolatey falló o no disponible: $($_.Exception.Message)"
    }

    Write-Error "No se pudo instalar Python automáticamente. Instálalo manualmente desde https://www.python.org/downloads/ y vuelve a ejecutar el script."
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
        Write-Error "requirements.txt no encontrado en $scriptDir"
        exit 1
    }

    # Instalar/verificar Python
    $pythonCmd = Install-Python

    # Crear venv si falta
    $venvPath = Join-Path $scriptDir 'venv'
    if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
        Write-Stamp "Creando virtualenv en $venvPath ..."
        & $pythonCmd -m venv $venvPath
    }

    $venvPython = Join-Path $venvPath 'Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        Write-Error "No se creó correctamente el entorno virtual."
        exit 1
    }

    # Upgrade pip/setuptools/wheel dentro del venv
    Write-Stamp "Actualizando pip, setuptools y wheel..."
    & $venvPython -m pip install --upgrade pip setuptools wheel --disable-pip-version-check

    # Intento 1: instalar requirements preferiendo binarios
    Write-Stamp "Instalando paquetes desde requirements.txt (prefer-binary) ..."
    $installSucceeded = $false
    try {
        & $venvPython -m pip install -r $requirementsFile --prefer-binary --disable-pip-version-check
        $installSucceeded = $true
    } catch {
        Write-Stamp "Primera instalación falló: $($_.Exception.Message)"
    }

    # Si falló y el fallo parece por compilación nativa, instalar Build Tools y reintentar
    if (-not $installSucceeded) {
        Write-Stamp "Instalación fallida. Detectando si es necesario instalar Build Tools (compilación nativa)..."
        # Heurística simple: si falta vswhere.exe o meson, instalamos Build Tools
        $needBuild = $false
        if (-not (Command-Exists "vswhere.exe")) { $needBuild = $true }
        if (-not $needBuild) {
            # grep logs? no tenemos logs aquí, mejor intentar instalar Build Tools si hubo fallo
            $needBuild = $true
        }

        if ($needBuild) {
            Install-BuildTools
            Write-Stamp "Reintentando instalación de requirements (sin prefer-binary para permitir builds si es necesario)..."
            try {
                & $venvPython -m pip install -r $requirementsFile --disable-pip-version-check
                $installSucceeded = $true
            } catch {
                Write-Stamp "Reintento después de Build Tools falló: $($_.Exception.Message)"
            }
        }
    }

    if (-not $installSucceeded) {
        Write-Error "No se pudieron instalar todas las dependencias automáticamente. Revisa el output arriba. Puedes intentar manualmente:`n`t .\venv\Scripts\python.exe -m pip install -r requirements.txt"
        exit 1
    }

winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --accept-source-agreements --accept-package-agreements

    Write-Stamp "Instalación completada. Activa el venv con:`n`t .\\venv\\Scripts\\Activate.ps1"
}
finally {
    Pop-Location
}


