# 에란겔 자기장 데이터 수집 설계

## 목표

1경기를 하나의 시퀀스로 저장합니다.

```text
1 Match = 1 Plane Route + Phase 1 Circle + ... + Phase 9 Circle
```

초기 모델은 플레이어 위치를 제외하고 다음 데이터만 사용합니다.

- 비행기 경로
- 1~9페이즈 자기장 중심
- 1~9페이즈 자기장 반경

## 테이블

### matches

| 컬럼 | 설명 |
| --- | --- |
| match_id | 경기 ID |
| shard_id | 플랫폼 shard |
| map_name | 맵 이름 |
| game_mode | 게임 모드 |
| created_at | 경기 생성 시간 |
| telemetry_url | 텔레메트리 URL |

### plane_routes

| 컬럼 | 설명 |
| --- | --- |
| match_id | 경기 ID |
| start_x/start_y | 추정 비행기 시작 좌표 |
| end_x/end_y | 추정 비행기 종료 좌표 |
| angle | 경로 각도 |
| route_group | 경로 그룹 |
| confidence | 추정 신뢰도 |

### circles

| 컬럼 | 설명 |
| --- | --- |
| match_id | 경기 ID |
| phase | 자기장 페이즈 |
| center_x/center_y | 중심 좌표 |
| radius | 반경 |
| timestamp | 텔레메트리 이벤트 시각 |
| elapsed_time | 경기 경과 시간 |

## 구현 메모

- 원본 match/telemetry JSON은 `data/raw`에 캐시합니다.
- API 요청 제한은 기본 10 RPM으로 둡니다.
- `LogGameStatePeriodic.common.isGame` 값이 정수인 `1.0 ~ 9.0` 상태를 새 안전구역 생성 시점으로 봅니다.
- `poisonGasWarningPosition`이 있으면 다음 안전구역 후보로 우선 사용하고, 없으면 `safetyZonePosition`을 사용합니다.
