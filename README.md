# PUBG Erangel Circle Predictor

PUBG 에란겔 경기의 비행기 경로와 1~9페이즈 자기장 시퀀스를 수집하는 1차 데이터셋 구축 프로젝트입니다.

## 현재 수집 범위

- Match: `match_id`, `map_name`, `created_at`, `game_mode`, `shard_id`
- Plane Route: 시작/끝 좌표, 각도, 수동/자동 route group
- Circle: 1~9페이즈 중심 좌표, 반경, 생성 시점
- 분석 파생값: 페이즈 이동 벡터, 거리, 각도, 축소율

공식 API에는 비행기 경로 전용 필드가 없어서, 텔레메트리의 초반 `LogPlayerPosition` 좌표를 이용해 비행기 경로를 추정합니다.

## 설치

```powershell
cd F:\coding\CircleTrain
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

`.env`에 PUBG API 키를 넣습니다.

```text
PUBG_API_KEY=...
PUBG_SHARD=steam
PUBG_REQUESTS_PER_MINUTE=10
```

## 데이터 수집

최근 샘플 매치에서 에란겔만 필터링해 수집합니다.

```powershell
py -m circle_train.collector collect-samples --limit 20
```

기본값은 비행기 동선이 추정된 경기만 저장합니다. 즉 `plane_routes`가 없는 경기는 스킵됩니다.

특정 매치 ID를 직접 수집합니다.

```powershell
py -m circle_train.collector collect-match --match-id MATCH_ID
```

CSV로 내보냅니다.

```powershell
py -m circle_train.collector export
```

## 1차 분석

```powershell
py -m circle_train.analysis vectors
py -m circle_train.analysis route-summary
```

결과는 `data/processed`에 저장됩니다.

## 데이터 기준

PUBG 텔레메트리 좌표는 센티미터 단위이며, 에란겔 X/Y 범위는 `0 ~ 816000`입니다. 이 프로젝트는 원본 좌표를 그대로 저장하고, 분석 시 필요하면 km 단위로 변환합니다.
