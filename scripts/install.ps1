# AgentSchool Windows Installer (PowerShell)
# Usage: iex (Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/HKUDS/AgentSchool/main/scripts/install.ps1')
#        or: powershell -ExecutionPolicy Bypass -File scripts/install.ps1

param(
    [switch]$FromSource,
    [switch]$Help
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
function Write-Info { Write-Host "[INFO]  $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[OK]    $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }
function Write-Step { Write-Host ""; Write-Host "==>$args" -ForegroundColor Blue -BackgroundColor White }

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  ==============================" -ForegroundColor Cyan
Write-Host "    AgentSchool Installer" -ForegroundColor Cyan
Write-Host "    Windows Native Setup" -ForegroundColor Cyan
Write-Host "  ==============================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if ($Help) {
    Write-Host "Usage: .\install.ps1 [-FromSource]"
    Write-Host ""
    Write-Host "  -FromSource    Clone from GitHub and install in editable mode"
    exit 0
}

# ---------------------------------------------------------------------------
# Step 1: Check PowerShell version
# ---------------------------------------------------------------------------
Write-Step "Checking PowerShell version"

if ($PSVersionTable.PSVersion.Major -lt 5) {
    Write-Error "PowerShell 5.1 or newer is required."
    Write-Host "  Please upgrade PowerShell or use PowerShell Core (pwsh):"
    Write-Host "    https://github.com/PowerShell/PowerShell"
    exit 1
}

Write-Success "PowerShell $($PSVersionTable.PSVersion) detected"

# ---------------------------------------------------------------------------
# Step 2: Check Python 3.10+
# ---------------------------------------------------------------------------
Write-Step "Checking Python version (3.10+ required)"

$PythonCmd = $null
$PythonCommands = @("python", "python3", "py")

foreach ($cmd in $PythonCommands) {
    $pyPath = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($pyPath) {
        $versionOutput = & $cmd --version 2>&1
        $versionMatch = $versionOutput -match "Python (\d+)\.(\d+)"
        if ($versionMatch) {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                $PythonCmd = $cmd
                break
            } elseif ($major -eq 3 -and $minor -lt 10) {
                Write-Warn "Python $major.$minor found but version 3.10+ is required"
            }
        }
    }
}

if (-not $PythonCmd) {
    Write-Error "Python 3.10+ not found."
    Write-Host ""
    Write-Host "  Please install Python 3.10 or newer:"
    Write-Host "    Download from: https://www.python.org/downloads/"
    Write-Host "    Or use winget: winget install Python.Python.3.12"
    Write-Host ""
    exit 1
}

$PyVersion = & $PythonCmd --version 2>&1
Write-Success "Found $PyVersion ($PythonCmd)"

# ---------------------------------------------------------------------------
# Step 3: Install AgentSchool
# ---------------------------------------------------------------------------
Write-Step "Installing AgentSchool"

$RepoUrl = "https://github.com/HKUDS/AgentSchool.git"
$InstallDir = "$env:USERPROFILE\.agentschool-src"
$VenvDir = "$env:USERPROFILE\.agentschool-venv"

# Create virtual environment
if (Test-Path $VenvDir) {
    Write-Info "Virtual environment already exists at $VenvDir"
} else {
    Write-Info "Creating virtual environment at $VenvDir..."
    & $PythonCmd -m venv $VenvDir
    if (-not (Test-Path $VenvDir)) {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
}

# Activate the venv
$ActivateScript = "$VenvDir\Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Error "Virtual environment activation script not found: $ActivateScript"
    exit 1
}

Write-Info "Activating virtual environment..."
& $ActivateScript

Write-Success "Virtual environment ready: $VenvDir"

# Install AgentSchool
if ($FromSource) {
    Write-Info "Mode: -FromSource (git clone + pip install -e .)"
    
    $GitPath = Get-Command git -ErrorAction SilentlyContinue
    if (-not $GitPath) {
        Write-Error "git is required for -FromSource installation."
        Write-Host "  Install git and retry:"
        Write-Host "    winget install Git.Git"
        Write-Host "    Or download from: https://git-scm.com/download/win"
        exit 1
    }
    
    if (Test-Path "$InstallDir\.git") {
        Write-Info "Source directory exists, pulling latest changes..."
        Push-Location $InstallDir
        git pull --ff-only
        Pop-Location
    } else {
        Write-Info "Cloning AgentSchool into $InstallDir..."
        git clone $RepoUrl $InstallDir
        if (-not (Test-Path $InstallDir)) {
            Write-Error "Failed to clone repository"
            exit 1
        }
    }
    
    Write-Info "Installing in editable mode (pip install -e .)..."
    pip install -e $InstallDir --quiet
} else {
    Write-Info "Mode: pip install agentschool-ai"
    pip install agentschool-ai --quiet --upgrade
}

Write-Success "AgentSchool package installed"

# ---------------------------------------------------------------------------
# Step 4: Create AgentSchool config directory
# ---------------------------------------------------------------------------
Write-Step "Setting up AgentSchool config directory"

$ConfigDir = "$env:USERPROFILE\.agentschool"
$SkillsDir = "$ConfigDir\skills"
$PluginsDir = "$ConfigDir\plugins"

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
New-Item -ItemType Directory -Force -Path $SkillsDir | Out-Null
New-Item -ItemType Directory -Force -Path $PluginsDir | Out-Null

Write-Success "Config directory ready: ~/.agentschool/"

# ---------------------------------------------------------------------------
# Step 5: Add to PATH (Windows environment variable)
# ---------------------------------------------------------------------------
Write-Step "Setting up PATH integration"

$VenvBinDir = "$VenvDir\Scripts"
$CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

if ($CurrentPath -like "*$VenvBinDir*") {
    Write-Info "PATH already contains $VenvBinDir"
} else {
    Write-Info "Adding $VenvBinDir to user PATH..."
    $NewPath = "$VenvBinDir;$CurrentPath"
    [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
    Write-Success "Added $VenvBinDir to PATH"
    Write-Warn "You may need to restart your terminal or log out/log in for PATH changes to take effect."
}

# ---------------------------------------------------------------------------
# Step 6: Verify installation
# ---------------------------------------------------------------------------
Write-Step "Verifying installation"

$AgentSchoolPath = "$VenvBinDir\agentschool.exe"

if (Test-Path $AgentSchoolPath) {
    $AgentSchoolVersion = & $AgentSchoolPath --version 2>&1
    Write-Success "Installation successful!"
    Write-Host ""
    Write-Host "  agentschool is ready: $AgentSchoolVersion" -ForegroundColor Green
} else {
    # Try module execution
    $ModuleVersion = python -m agentschool --version 2>&1
    if ($ModuleVersion) {
        Write-Warn "Launcher commands not yet available on PATH. Run via: python -m agentschool"
        Write-Host "  Version: $ModuleVersion"
    } else {
        Write-Warn "Could not verify launcher commands. The package may need a PATH update."
        Write-Host "  Try: python -m agentschool --version"
    }
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "AgentSchool is installed!" -ForegroundColor Green -BackgroundColor White
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    1. Restart terminal, or run: refreshenv (if using Chocolatey)"
Write-Host "       Or manually refresh: `$env:PATH = [System.Environment]::GetEnvironmentVariable('PATH','User')"
Write-Host "    2. Set your API key:        `$env:ANTHROPIC_API_KEY = 'your_key'"
Write-Host "    3. Launch (PowerShell):     agentschool"
Write-Host "    4. Docs:                    https://github.com/HKUDS/AgentSchool"
Write-Host ""
