# Automation

## Files

- `dist/CircleTrainAutoCollector.exe`
  - Double-click launcher.
  - Runs one automated collection pass.
  - Shows a Windows message box when the run finishes.

- `scripts/run_auto_collect_once.ps1`
  - Runs one automated collection pass from PowerShell.

- `scripts/run_auto_collect_daily.ps1`
  - Runs collection every 24 hours while the script remains open.

- `scripts/register_daily_task.ps1`
  - Registers a Windows Task Scheduler job named `CircleTrainDailyCollector`.

## Default EXE Behavior

The exe runs:

```powershell
py -m circle_train.auto_collect --target-new 1000 --days 14 --shards steam,kakao --min-circles 2 --no-notify
```

This means:

- scan recent 14 UTC days
- scan `steam` and `kakao`
- require Erangel
- require inferred plane route
- require at least 2 circle phases
- export CSV files
- export phase vector and route summary analysis
- show final Windows alert from the launcher

## Strict P1-P9 Mode

Use this when you want only full 9-phase matches:

```powershell
dist\CircleTrainAutoCollector.exe --target-new 100 --days 14 --shards steam,kakao --strict-full-sequence --no-notify
```

## Register Daily Collection

Build the exe first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
```

Register the scheduled task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1
```

After this, Windows runs collection daily at 2:00 AM.

## Notes

- `.env` must exist in the project root.
- `.env` is ignored by git and should not be committed.
- The PUBG API key is read from `PUBG_API_KEY`.
- Request rate is controlled by `PUBG_REQUESTS_PER_MINUTE`.
