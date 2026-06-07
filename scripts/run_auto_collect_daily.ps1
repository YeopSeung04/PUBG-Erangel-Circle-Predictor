$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

py -m circle_train.auto_collect `
  --target-new 1000 `
  --days 14 `
  --shards steam,kakao `
  --min-circles 2 `
  --repeat `
  --interval-hours 24
