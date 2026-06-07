$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ExePath = Join-Path $ProjectRoot "dist\CircleTrainAutoCollector.exe"

if (-not (Test-Path $ExePath)) {
  throw "Build the exe first: scripts\build_exe.ps1"
}

$Action = New-ScheduledTaskAction -Execute $ExePath -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask `
  -TaskName "CircleTrainDailyCollector" `
  -Action $Action `
  -Trigger $Trigger `
  -Settings $Settings `
  -Description "Collect PUBG Erangel plane route and circle data daily." `
  -Force

Write-Host "Registered daily task: CircleTrainDailyCollector"
