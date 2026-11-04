# 혐오체인 제출 자료 · 입력 데이터 샘플과 모델 메타데이터

## 1. 입력 데이터 샘플 — `input_sample.csv`

| 항목 | 내용 |
|---|---|
| 크기 | 999,378 bytes (1MB 이하), 6,347행, UTF-8 CSV (Excel용 BOM 포함) |
| 출처 | K-HATERS 4,210행 · K-MHaS 1,959행 · BEEP 178행 (정제 코퍼스 311,661행에서 추출) |
| 추출 방식 | 출처 × 라벨 비율을 유지한 층화 무작위 추출, seed 20261105 (`make_sample.py`) |
| 라벨 분포 | 정상 2,270행 · 유해 4,077행 (원본 유해 비율과 비슷한 64%) |
| 제외한 출처 | NIKL · UnSmile · KOLD — 재배포가 허용되지 않거나 라이선스가 명시되지 않음 |
| 라이선스 | K-HATERS CC-BY-4.0, K-MHaS CC-BY-SA-4.0, BEEP CC-BY-SA-4.0 (출처 표기 필요) |

| 열 | 의미 |
|---|---|
| `sample_id` | 샘플 번호 |
| `text` | **모델 입력**: 한국어 문장 (1–1,000자, 한글 포함) |
| `label` / `label_id` | 정답: 정상 `clean` = 0, 유해 `harmful` = 1 (출처 라벨을 이진 변환) |
| `source` | 원래 데이터셋 |
| `split` | 학습 `train` / 검증 `validation` / 평가 `test` (연결 그룹 단위 80/10/10 분할 기준) |
| `length` | 글자 수 |

## 2. 모델 메타데이터 — `model_metadata.json`

| 항목 | 모델 A · 6종 이진 분류 | 모델 B · K-HATERS 점수 |
|---|---|---|
| 과제 | 이진 분류 (정상 0 / 유해 1) | 3등급 분류(0·1·2) → 0–100 점수 3개 |
| 입력 | 한국어 문장 (str, 1–1,000자, 한글 포함) | 같은 입력 |
| 출력 | 판정 정상·유해·보류 + 점수 0–100 | Hate · 공격성 · 욕설 점수 0–100 |
| 학습 데이터 | 6종 통합 311,661행 (train 249,732 / val 30,580 / test 31,349) | K-HATERS Ratings (172,097 / 9,997 / 9,997) |
| 정답 | 출처 라벨을 정상/유해로 변환 | Hate = 집단 평점 7개 최댓값, 공격성 = 13개 최댓값, 욕설 = 욕설 평점 |
| 속성 | 문자 2–5gram TF-IDF 200,000개 (sublinear tf, L2 정규화) | 문자 2–5gram TF-IDF 100,000개 |
| 알고리즘 | LogisticRegression (liblinear, L2, C=1, class_weight=balanced) | LogisticRegression × 3 (lbfgs, 3등급 softmax) |
| 수식 | P = σ(wᵀx + b), 점수 = 100P | P(k) = softmax(z_k / T), 점수 = 50 × (P(1) + 2·P(2)) |
| 판정·보정 | P ≥ 0.43 → 유해 (validation Macro-F1로 선택), 인식 조각 0개 → 보류 | 온도 T = 1.0 (validation NLL로 선택), 판정 기준 없음 |
| test 성능 | Macro-F1 0.7784 · 유해 재현율 83.52% · 정상 오탐률 27.82% · 정확도 79.33% | MAE Hate 19.77 · 공격성 25.76 · 욕설 15.02 (0–100점) |
| 기준선 | 다수 클래스(전부 유해) Macro-F1 0.3865 | train 평균값 MAE 32.44 / 35.22 / 23.16 |
| 모델 파일 | `char_2_5_200000.joblib` 9.4MB, SHA-256 `68cc4e74…` | `khaters_score_v2/model.joblib` 11.0MB, SHA-256 `29720d7b…` |
| 학습일 | 2026-10-10 (학습 116초) | 2026-10-26 |
| 실행 환경 | Python 3.11.14, scikit-learn 1.9.0, CPU 단일 서버, Flask 웹 | 같음 |
| 한계 | 같은 6종 안의 내부 평가 · 부정문·인용 문맥 미반영 · 짧은 입력은 조각 하나가 좌우 | 점수는 확률·심각도가 아님 · 약한 혐오(1등급) 누락 많음 |

전체 값(해시 전체, 혼동행렬, 하이퍼파라미터, validation 지표)은 `model_metadata.json`에 있다. 모델 파일 자체는 용량과 라이선스 때문에 제출물에 포함하지 않는다.
