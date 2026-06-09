# PUBG Erangel Circle Predictor

PUBG 에란겔 경기의 비행기 동선과 자기장 흐름을 수집해서, 이후 자기장 예측 모델을 만들기 위한 데이터 수집 프로젝트입니다.

초기 목표는 AI 모델 학습이 아니라 다음 데이터셋을 안정적으로 누적하는 것입니다.

```text
비행기 동선 + 자기장 페이즈 시퀀스
```

## 수집 데이터

- Match: `match_id`, `map_name`, `created_at`, `game_mode`, `shard_id`
- Plane Route: 시작 좌표, 종료 좌표, 각도, route group
- Circle: 페이즈별 중심 좌표, 반경, 생성 시점
- Analysis: 페이즈 이동 벡터, 이동 거리, 이동 각도, 축소율

공식 PUBG API에는 비행기 동선 전용 필드가 없습니다. 그래서 텔레메트리의 초반 `LogPlayerPosition` 좌표를 이용해 비행기 경로를 추정합니다.

## 데이터 단위

전체 시퀀스 데이터:

```text
1 Match
= 1 Plane Route
+ Phase 1 Circle
+ ...
+ Phase 9 Circle
```

학습용 transition 데이터:

```text
Plane Route + P1 -> P2
Plane Route + P1 + P2 -> P3
...
P8 -> P9
```

P1~P9 전체가 있는 경기만 사용하면 통과율이 낮습니다. 그래서 실제 예측 학습에서는 `min-circles=2` 이상인 경기를 모아 phase transition row를 많이 확보하는 방식이 더 현실적입니다.

## 현재 누적 기준

현재 기본 수집 조건:

```text
에란겔/Baltic_Main
+ 비행기 동선 추정 성공
+ 최소 2개 이상 자기장 페이즈
```

엄격한 검증용 데이터는 `--strict-full-sequence` 옵션으로 P1~P9 전체 경기만 모을 수 있습니다.

## 예상 하루 수집량

GitHub Actions 기본 설정 기준:

```text
schedule: 하루 4회, 02:00/08:00/14:00/20:00 KST
target-new: 300/run, 하루 최대 목표 1200
days: 14
shards: steam,kakao
min-circles: 2
비행기 동선 필수
에란겔만
timeout: 6시간/run
```

현실적인 예상:

```text
예측용 match 수집 목표: 하루 1000개 이상
기대 phase transition 학습 row: 하루 약 8,000개 이상
```

현재 관측 기준으로 match 1개당 평균 약 8개의 transition row가 나옵니다.

예:

```text
1000 matches/day x 8 transitions
= 약 8,000 training rows/day
```

목표별 예상:

```text
4만 transition row: 목표 속도 기준 약 5일
4만 full P1~P9 match: 훨씬 오래 걸림
```

## 설치

```powershell
cd F:\coding\CircleTrain
py -m venv .venv
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

## 수동 수집

최근 sample에서 에란겔 경기만 필터링해 수집합니다.

```powershell
py -m circle_train.collector collect-samples --limit 20
```

최근 14일 sample window를 `steam,kakao`에서 훑습니다.

```powershell
py -m circle_train.collector collect-history --limit 1000 --days 14 --shards steam,kakao --min-circles 2 --quiet-skip
```

P1~P9 전체 경기만 모으려면:

```powershell
py -m circle_train.collector collect-history --limit 100 --days 14 --shards steam,kakao --min-circles 9 --quiet-skip
```

CSV export:

```powershell
py -m circle_train.collector export
```

## 분석

```powershell
py -m circle_train.analysis vectors
py -m circle_train.analysis route-summary
```

결과는 `data/processed`에 저장됩니다.

### 1000개/day 목표 기준

현재 GitHub Actions schedule은 하루 4회 실행됩니다.

```text
02:00 KST -> 최대 300개
08:00 KST -> 최대 300개
14:00 KST -> 최대 300개
20:00 KST -> 최대 300개
하루 목표 -> 최대 1200개
```

단, 이 값은 목표치입니다. 실제 수집량은 그날 공개 sample에 에란겔 경기와 비행기 동선 추정 가능 경기가 얼마나 있는지에 따라 달라집니다.

GitHub-hosted Actions는 private repo에서 월별 실행 시간 제한이 있을 수 있습니다. 안정적으로 1000개/day 이상을 계속 모으려면 장기적으로는 self-hosted runner 또는 로컬 exe 자동 수집을 같이 쓰는 편이 안전합니다.

## GitHub Actions 자동 수집

워크플로:

```text
.github/workflows/collect-data.yml
```

동작:

- 매일 02:00, 08:00, 14:00, 20:00 KST 실행
- `steam,kakao` shard 수집
- GitHub Secret `PUBG_API_KEY` 사용
- raw telemetry는 커밋하지 않음
- 누적 SQLite/CSV는 private repo에 커밋

누적 파일:

```text
data/circle_train.sqlite
data/processed/*.csv
```

수동 실행은 GitHub Actions 탭에서 `Collect PUBG Erangel Circle Data` workflow를 실행하면 됩니다.

## Windows exe 자동 수집

로컬 PC에서 실행하려면:

```powershell
dist\PUBGErangelCircleCollector.exe
```

매일 로컬 작업 스케줄러에 등록하려면:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1
```

## 좌표 기준

PUBG 텔레메트리 좌표는 센티미터 단위입니다. 에란겔 X/Y 범위는 대략 `0 ~ 816000`입니다. 이 프로젝트는 원본 좌표를 그대로 저장하고, 분석이나 모델 학습 시 필요한 경우 km 단위로 변환합니다.

<!-- COLLECTION_LOG_START -->
## 수집 로그

총 수집 데이터: 2432개

- 2026/0610: 300개 데이터
- 2026/0609: 300개 데이터
- 2026/0608: 300개 데이터
- 2026/0607: 2개 데이터
<!-- COLLECTION_LOG_END -->
