param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [string]$User = "ubuntu",
    [string]$RemoteRepoDir = "~/ECESSP/ecessp-ml"
)

$ErrorActionPreference = "Stop"

$files = @(
    "ecessp-ml/data/processed/material_catalog.csv",
    "ecessp-ml/data/processed/batteries_parsed.csv",
    "ecessp-ml/data/processed/batteries_parsed_curated.csv",
    "ecessp-ml/data/processed/batteries_ml_curated.csv",
    "ecessp-ml/graphs/masked_battery_graph_normalized_v2.pt",
    "ecessp-ml/graphs/battery_hetero_graph_v1.pt",
    "ecessp-ml/reports/training_summary.json",
    "ecessp-ml/reports/final_family_ensemble_manifest.json",
    "ecessp-ml/reports/three_model_ensemble_manifest.json",
    "ecessp-ml/reports/active_learning_queue.jsonl"
)

$directories = @(
    "ecessp-ml/reports/models"
)

$destinationMap = @{
    "ecessp-ml/data/processed/material_catalog.csv" = "$User@${HostName}:$RemoteRepoDir/data/processed/"
    "ecessp-ml/data/processed/batteries_parsed.csv" = "$User@${HostName}:$RemoteRepoDir/data/processed/"
    "ecessp-ml/data/processed/batteries_parsed_curated.csv" = "$User@${HostName}:$RemoteRepoDir/data/processed/"
    "ecessp-ml/data/processed/batteries_ml_curated.csv" = "$User@${HostName}:$RemoteRepoDir/data/processed/"
    "ecessp-ml/graphs/masked_battery_graph_normalized_v2.pt" = "$User@${HostName}:$RemoteRepoDir/graphs/"
    "ecessp-ml/graphs/battery_hetero_graph_v1.pt" = "$User@${HostName}:$RemoteRepoDir/graphs/"
    "ecessp-ml/reports/training_summary.json" = "$User@${HostName}:$RemoteRepoDir/reports/"
    "ecessp-ml/reports/final_family_ensemble_manifest.json" = "$User@${HostName}:$RemoteRepoDir/reports/"
    "ecessp-ml/reports/three_model_ensemble_manifest.json" = "$User@${HostName}:$RemoteRepoDir/reports/"
    "ecessp-ml/reports/active_learning_queue.jsonl" = "$User@${HostName}:$RemoteRepoDir/reports/"
}

Write-Host "Creating target directories on $HostName"
ssh "$User@$HostName" "mkdir -p $RemoteRepoDir/data/processed $RemoteRepoDir/graphs $RemoteRepoDir/reports/models"

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        throw "Missing required file: $file"
    }

    Write-Host "Uploading $file"
    scp $file $destinationMap[$file]
}

foreach ($directory in $directories) {
    if (-not (Test-Path $directory)) {
        throw "Missing required directory: $directory"
    }

    Write-Host "Uploading $directory"
    scp -r $directory "$User@${HostName}:$RemoteRepoDir/reports/"
}

Write-Host "Runtime asset upload complete."
