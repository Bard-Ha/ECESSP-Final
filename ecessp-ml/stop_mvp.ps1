param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5000
)

function Stop-ByPort([int]$Port) {
  $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if ($conns) {
    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
      try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        Write-Host "Stopped PID $procId on port $Port"
      } catch {
        Write-Warning "Could not stop PID $procId on port $($Port): $($_.Exception.Message)"
      }
    }
  } else {
    Write-Host "No listener on port $Port"
  }
}

Stop-ByPort -Port $BackendPort
Stop-ByPort -Port $FrontendPort

# Optional cleanup for stale dev servers started from PowerShell windows
Get-CimInstance Win32_Process |
  Where-Object {
    ($_.Name -match "python|node") -and
    ($_.CommandLine -match "uvicorn|start_backend\.ps1|tsx server/index\.ts|npm run dev")
  } |
  ForEach-Object {
    try {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
      Write-Host "Stopped stale process $($_.ProcessId)"
    } catch {}
  }

Write-Host "MVP services stop routine complete."
