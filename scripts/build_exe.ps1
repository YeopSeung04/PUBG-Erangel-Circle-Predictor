$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LauncherProject = Join-Path $ProjectRoot "tools\CircleTrainAutoCollector\CircleTrainAutoCollector.csproj"
$OutputDir = Join-Path $ProjectRoot "dist"

dotnet publish $LauncherProject `
  -c Release `
  -r win-x64 `
  --self-contained false `
  -p:PublishSingleFile=true `
  -o $OutputDir

Write-Host "Built: $OutputDir\PUBGErangelCircleCollector.exe"
