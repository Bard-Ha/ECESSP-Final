# start_backend.ps1
# Start the ML backend from this repo using a local virtualenv Python.

param(
  [int]$Port = 8000,
  [string]$BindHost = "127.0.0.1",
  [switch]$Reload
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoDir = Split-Path -Parent $scriptDir
$pyCandidates = @(
  (Join-Path $scriptDir ".venv\Scripts\python.exe"),
  (Join-Path $repoDir ".venv\Scripts\python.exe")
)

$py = $pyCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-Not $py) {
  Write-Error "Python executable not found. Checked: $($pyCandidates -join ', ')"
  exit 1
}

Write-Host "Using Python: $py"
Write-Host "Starting backend (uvicorn) on http://$BindHost`:$Port (Ctrl+C to stop)"

Push-Location $scriptDir
try {
  $args = @("-m", "uvicorn", "backend.main:app", "--host", $BindHost, "--port", "$Port")
  if ($Reload) {
    $args += "--reload"
  }

  & $py @args 2>&1 | ForEach-Object { Write-Output $_ }
  $exit = $LASTEXITCODE

  if ($exit -ne 0) {
    Write-Error "Backend exited with code $exit"
    exit $exit
  }
} catch {
  Write-Error "Unhandled error while starting backend:`n$($_.Exception)"
  exit 1
} finally {
  Pop-Location
}
