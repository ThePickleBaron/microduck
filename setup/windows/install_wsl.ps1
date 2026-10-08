# Install WSL2 + Ubuntu 24.04 on Windows for the Microduck pre-training project.
#
# Run in an ELEVATED PowerShell (right-click > Run as administrator):
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup\windows\install_wsl.ps1
#
# GPU training (desktop, RTX 3090): CUDA inside WSL2 uses the normal Windows
# NVIDIA driver. Do NOT install a Linux NVIDIA driver inside Ubuntu.

$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "Please re-run this script from an elevated (Administrator) PowerShell." -ForegroundColor Yellow
    exit 1
}

Step "Windows build"
$build = [int](Get-CimInstance Win32_OperatingSystem).BuildNumber
Write-Host "Build $build"
if ($build -lt 19044) {
    Write-Host "WSL2 GPU support needs Windows 10 21H2 (build 19044) or Windows 11." -ForegroundColor Yellow
}

Step "NVIDIA driver (only matters on the training desktop)"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
} else {
    Write-Host "nvidia-smi not found: fine on the laptop (evaluation only)." -ForegroundColor Yellow
    Write-Host "On the desktop, install the current Game Ready or Studio driver from nvidia.com first."
}

Step "Installing / updating WSL2 with Ubuntu 24.04"
wsl --update
$distros = (wsl --list --quiet) -join " "
if ($distros -match "Ubuntu-24.04") {
    Write-Host "Ubuntu-24.04 already installed."
} else {
    wsl --install -d Ubuntu-24.04
    Write-Host "If Windows asks for a restart, reboot, then open 'Ubuntu 24.04' from the Start menu"
    Write-Host "once to create your Linux user name and password."
}
wsl --set-default Ubuntu-24.04

Write-Host @"

Next, inside Ubuntu (Start menu > Ubuntu 24.04):

  git clone https://github.com/ThePickleBaron/microduck.git ~/microduck-pretraining
  cd ~/microduck-pretraining
  bash setup/setup.sh

On the desktop, 'uv run md-check' should then report the RTX 3090 under
'CUDA GPU for training'. Details: docs/windows_wsl_setup.md
"@ -ForegroundColor Green
