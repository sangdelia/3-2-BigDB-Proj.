# 혐오체인 제출 자료 · 입력 데이터 샘플과 모델 메타데이터

## 1. 입력 데이터 샘플 — `input_sample.csv`

| 항목 | 내용 |
|---|---|
| 크기 | 999,999 bytes (1MB 이하), 5,946행, UTF-8 CSV (Excel용 BOM 포함) |
| 출처 | 모델 A 코퍼스(정제 311,661행)에서 5,520행: K-HATERS 3,661 · K-MHaS 1,701 · BEEP 158 / 모델 B 데이터(K-HATERS Ratings)의 validation·test에서 426행 |
| 추출 방식 | A: 출처 × 라벨 비율 층화, B validation·test: 공격성 등급 비율 층화. 무작위 seed 20261105 (`make_sample.py`) |
| 모델 A 구간 (`split`) | train 4,420 · validation 566 · test 534 (나머지 426행은 A 데이터에 없는 B 전용 행 → 빈칸) |
| 모델 B 구간 (`b_split`) | K-HATERS 4,087행 모두 정답 등급 포함: train 3,661 · validation 213 · test 213 (B 공식 분할 비율 172,097 : 9,997 : 9,997). 다른 출처는 빈칸 |
| 제외한 출처 | NIKL · UnSmile · KOLD — 재배포가 허용되지 않거나 라이선스가 명시되지 않음 |
| 라이선스 | K-HATERS CC-BY-4.0, K-MHaS CC-BY-SA-4.0, BEEP CC-BY-SA-4.0 (출처 표기 필요) |

| 열 | 의미 |
|---|---|
| `sample_id` | 샘플 번호 |
| `text` | **모델 입력**: 한국어 문장 (1–1,000자, 한글 포함) |
| `label` / `label_id` | 정답: 정상 `clean` = 0, 유해 `harmful` = 1 (출처 라벨을 이진 변환) |
| `source` | 원래 데이터셋 |
| `split` | **모델 A** 기준 학습 `train` / 검증 `validation` / 평가 `test` (연결 그룹 단위 80/10/10). 빈칸이면 A 데이터에 없는 B 전용 행 |
| `length` | 글자 수 |
| `b_hate` | **모델 B 정답** Hate 등급 0·1·2 = K-HATERS 원본 집단 평점 7개(성별·연령·장애·직업·인종·종교·정치) 중 최댓값 |
| `b_abuse` | **모델 B 정답** 공격성 등급 0·1·2 = 원본 평점 13개 전체 중 최댓값 |
| `b_swear` | **모델 B 정답** 욕설 등급 0·1·2 = 원본 욕설 평점 |
| `b_split` | **모델 B** 기준 `train` / `validation` / `test` (K-HATERS 공식 분할) |

모델 B는 같은 `text`를 입력받아 세 점수를 0–100으로 낸다. 정답 등급 g를 점수로 비교할 때는 50 × g(0·50·100)를 쓴다. 모델 A의 K-HATERS 데이터는 K-HATERS 공식 train 파일에서 왔다. 그래서 A 코퍼스에서 뽑은 K-HATERS 행은 모두 B의 train이고, B의 validation·test 행은 B 데이터에서 따로 뽑아 넣었다. 이 426행은 A 데이터에 없으므로 `split`이 빈칸이고, `label`은 공격성 등급으로 정했다(0 → 정상, 1 이상 → 유해). A 코퍼스에서 뽑은 K-HATERS 행에서도 이 대응은 모두 성립한다(정상 = `b_abuse` 0, 유해 = `b_abuse` ≥ 1).

| 사용 | 행 고르기 |
|---|---|
| 모델 A 평가 | `split` = `test` (534행) |
| 모델 B 평가 | `b_split` = `test` (213행), 정답 `b_hate`·`b_abuse`·`b_swear` |

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
