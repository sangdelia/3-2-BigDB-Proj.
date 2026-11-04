# 중간발표 피드백 보완 실험 (2026-11-04)

모두 `/home/kyu/BigDB_Proj`(데이터·`.venv`·모델 파일이 있는 체크아웃)에서 실행한다. 서비스 모델과 기존 결과 파일은 바꾸지 않는다.

| 스크립트 | 내용 | 결과 |
|---|---|---|
| `learning_curve.py` | 모델 A 설정(2–5gram·20만·클래스 가중치) 그대로 train만 5/10/25/50/100%로 줄여 재학습. validation·test 고정. 다수 클래스 기준선도 계산 | `reports/postmidterm_20261105/majority.json`, `learning_curve/lc_*.json` |
| `fp_review.py` | 1차: 단일 조각 보류 규칙만. 실제 욕설(쓰레기·시발)까지 보류해 채택하지 않음 | `fp_review.json` |
| `fp_review2.py` | 2차: 경계 보류(R1) + 단일 조각 보류(R2). 처음 정한 기준(보류 ≤5%, 자동 판정 재현율 ≥ 기존)으로는 어떤 규칙도 선택되지 않음 | `fp_review2.json` |
| `fp_review3.py` | 3차: 기준을 "보류 ≤5%에서 validation Macro-F1 최대"로 바꿔 선택 → test 1회 평가 | `fp_review3.json` |
| `build_deck_v13.py` | 보완본 PPT 생성 스크립트 | `ppts/` |

```bash
cd /home/kyu/BigDB_Proj
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python <worktree>/experiments/learning_curve.py <out> 0.05 0.1 0.25 0.5 1.0
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python <worktree>/experiments/fp_review3.py <out>
```

## 주요 결과 (test)

- 다수 클래스(전부 유해) Macro-F1 0.3865 → 모델 A 0.7784
- 학습 곡선 test Macro-F1: 5% 0.7262 · 10% 0.7398 · 25% 0.7582 · 50% 0.7698 · 100% 0.7784
- 보류 규칙(R1: 43 ≤ 점수 < 48, R2: 조각 ≤ 2개·최대 비중 ≥ 60%·점수 < 80): 보류 4.79%, 자동 판정 정상 오탐률 27.86% → 23.64%, 유해 재현율 83.54% → 82.80%, Macro-F1 0.7783 → 0.7920. 보류 1,501건 중 844건은 실제 유해.
- 선택 기준을 결과를 본 뒤 한 번 바꿨으므로(2차 → 3차) 독립 검증으로 보지 않는다. 웹 서비스에는 반영하지 않았다.
