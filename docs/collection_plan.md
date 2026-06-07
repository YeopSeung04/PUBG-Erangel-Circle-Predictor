# PUBG Erangel Circle Prediction Collection Plan

## Why 40,000 Matches Matter

4 matches are only a pipeline test. A model that predicts Erangel circle flow needs tens of thousands of examples because the input is sparse and highly stochastic.

Target tiers:

- `100~500`: parser validation and visualization only
- `1,000~5,000`: basic route-group statistics
- `10,000+`: usable transition probability estimates
- `40,000+`: reasonable first ML training target

## Current Strict Dataset Rule

The production dataset currently saves only matches that satisfy all conditions:

```text
Erangel/Baltic_Main
+ inferred plane route
+ phase 1~9 circles
```

This is intentionally strict, but the yield is low because many public sample matches end before phase 9.

## Practical Collection Strategy

Use two datasets:

1. `full_sequence`
   - Requires phase 1~9.
   - Used for final sequence evaluation.

2. `transition_sequence`
   - Requires plane route and at least 2 phases.
   - Used for phase-to-phase learning:

```text
plane route + P1 -> P2
plane route + P1 + P2 -> P3
...
P8 -> P9
```

This gives far more training rows than only full 9-phase matches.

## Commands

Strict full-sequence collection:

```powershell
py -m circle_train.collector collect-history --limit 1000 --days 14 --shards steam,kakao --quiet-skip
```

Transition-oriented collection:

```powershell
py -m circle_train.collector collect-history --limit 1000 --days 14 --shards steam,kakao --min-circles 2 --quiet-skip
```

Refresh exports:

```powershell
py -m circle_train.collector export
py -m circle_train.analysis vectors
py -m circle_train.analysis route-summary
```

## Scaling Reality

At 10 RPM, this cannot become 40,000 full-sequence matches in a single run.

The collector should be run daily because PUBG public samples are time-window based. To reach 40,000, the project needs:

- daily scheduled collection
- both `steam` and `kakao` shards
- strict full-sequence data for validation
- looser transition data for training volume
- no fake or manually invented rows
