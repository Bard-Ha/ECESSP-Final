# run_mvp.ps1
# Launches backend and frontend in two PowerShell windows.

param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5000,
  [string]$BackendHost = "127.0.0.1"
)

$mlDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoDir = Split-Path -Parent $mlDir
$frontendDir = Join-Path $repoDir "ecessp-frontend"

if (-Not (Test-Path $frontendDir)) {
  Write-Error "Frontend directory not found: $frontendDir"
  exit 1
}

$backendUrl = "http://$BackendHost`:$BackendPort"

Write-Host "Starting backend in new PowerShell window..."
Start-Process -FilePath "powershell" -ArgumentList @(
  "-NoExit",
  "-NoProfile",
  "-Command",
  "cd '$mlDir'; .\start_backend.ps1 -BindHost '$BackendHost' -Port $BackendPort"
) -WindowStyle Normal

Start-Sleep -Seconds 2

Write-Host "Starting frontend in new PowerShell window..."
Start-Process -FilePath "powershell" -ArgumentList @(
  "-NoExit",
  "-NoProfile",
  "-Command",
  "cd '$frontendDir'; `$env:NODE_ENV='development'; `$env:PORT='$FrontendPort'; `$env:ECESSP_ML_URL='$backendUrl'; npm run dev"
) -WindowStyle Normal

Write-Host "Launched backend:  $backendUrl"
Write-Host "Launched frontend: http://localhost:$FrontendPort"
Write-Host "Frontend proxy uses ECESSP_ML_URL=$backendUrl"
