# ---- Setup for mujoco sim (Windows / PowerShell) ----
$ErrorActionPreference = "Stop"

# 1. Ensure uv is available.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ">> Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    $uvBin = "$env:USERPROFILE\.local\bin"

    # Refresh PATH for this session so uv works right now.
    $env:Path = "$uvBin;$env:Path"

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$uvBin*") {
        [Environment]::SetEnvironmentVariable("Path", "$uvBin;$userPath", "User")
       
    }
}

# 2. Create env + install from pyproject.toml, locked.
Write-Host ">> Syncing environment from pyproject.toml..."
uv sync

Write-Host ""
Write-Host ">> Done."
Write-Host ">> Run with:  uv run python robo-code/sim.py"