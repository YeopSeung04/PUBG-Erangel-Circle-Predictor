$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ExePath = Join-Path $ProjectRoot "dist\PUBGErangelCircleCollector.exe"

if (-not (Test-Path $ExePath)) {
  throw "Build the exe first: scripts\build_exe.ps1"
}

$Action = New-ScheduledTaskAction -Execute $ExePath -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask `
  -TaskName "PUBGErangelCircleDailyCollector" `
  -Action $Action `
  -Trigger $Trigger `
  -Settings $Settings `
  -Description "Collect PUBG Erangel plane route and circle prediction data daily." `
  -Force

Write-Host "Registered daily task: PUBGErangelCircleDailyCollector"
