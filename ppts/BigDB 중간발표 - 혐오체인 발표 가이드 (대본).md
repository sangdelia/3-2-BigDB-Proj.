# BigDB 중간발표 보완본 · 슬라이드별 요점 · 혐오체인

- 11/04 발표 피드백 반영본 (본문 21장 + 부록 8장). 5분 발표용 대본이 아니라 장별 설명 요점이다.
- 피드백 대응: 문제 정의(2장), 라벨→정답 변환(3장), 문장→숫자 속성(6장), HDFS–Spark 연동(9장), 모델 수식(11·12장), 오탐률 계산(14장), 기준선(15장), 데이터 규모 실험(16장), 오탐 보류 규칙(18장)

### 01. 혐오체인 : 
한국어 유해표현 분석 웹 페이지

혐오체인 팀의 ‘한국어 유해표현 분석 웹 페이지’ 중간 발표를 시작하겠습니다.

### 02. 문제 정의: 한국어 문장의 유해 여부 분류

피드백(무엇을 하려는지)에 대한 답. 이 프로젝트는 문장을 입력받아 정상/유해를 분류하는 지도학습이 주 과제이고, 같은 입력에 대해 Hate·공격성·욕설 점수를 예측하는 모델 B를 보조로 둔다. 모든 단계는 '정답 만들기 → 숫자 속성 만들기 → 학습 → 평가' 순서.

> 근거 파일(/home/kyu/BigDB_Proj/): docs/KOREAN_EXPERIMENT_PLAN.md · docs/KHATERS_HATE_SCORE.md

### 03. 6종 라벨 구조와 정답(0/1) 변환

피드백(정답 데이터를 만들어 학습)에 대한 답. 여섯 데이터의 라벨 체계를 하나의 정답 y∈{0,1}로 바꿨다. 같은 출처 안에서 '깨끗함'과 '혐오'가 동시에 붙는 등 모순되면 제외했다. 이 y가 모델 A의 학습·평가 정답이다.

> 근거 파일(/home/kyu/BigDB_Proj/): src/bigdb_mvp/data/arrow_sources.py · src/bigdb_mvp/data/remote_sources.py · src/bigdb_mvp/data/normalize.py · reports/eda_20261029/hadoop/aggregates.tsv

### 04. 데이터셋 EDA

데이터는 한국어 6종, 정제 후 31만 1,661행입니다. K-HATERS가 54.1%로 가장 크고 전체의 63%가 유해 라벨이라, 출처와 라벨 모두 불균형합니다. 필수값 결측은 없고, 비어 있는 선택 주석도 삭제하지 않고 보존했습니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/eda_20261029/report.json, reports/eda_20261029/sources.csv

### 05. 전처리 파이프라인과 빅데이터 적용 위치

전처리는 여섯 단계입니다. 공통 스키마 변환과 필수값 검사 뒤, 공백 정규화·충돌 제외·중복 병합으로 31만 1,661행을 만들고 연결 그룹 단위로 80/10/10 분할했습니다. 빅데이터 처리는 3단계 정제에 Spark, 5단계 저장·집계에 Hadoop을 적용했고, 둘 다 Python 결과와 일치하는지로 검증했습니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/eda_20261029/report.json, data/curated/korean_v2/manifest.json

### 06. 문장을 숫자 속성으로: 문자 n-gram TF-IDF

피드백(속성을 숫자로 정리해 표현)에 대한 답. 문장을 2–5글자 조각으로 나누고 TF-IDF로 숫자를 매긴 벡터가 속성 x다. 오른쪽 표는 실제 모델에서 이 문장의 조각별 값과 가중치이며, 합 5.733이 다음 장 수식의 wᵀx가 된다. 띄어쓰기(␣)도 조각에 포함된다.

> 근거 파일(/home/kyu/BigDB_Proj/): artifacts/korean_short_v1/char_2_5_200000.joblib · src/bigdb_mvp/experiments/classification.py

### 07. 같은 전처리 구간의 Python·Spark 비교

Spark는 같은 정제 구간을 단일 서버에서 Python과 비교했습니다. 세 규모 모두 출력 키 해시가 일치했고, 31만 행 규모에서만 Spark가 3.778초로 더 짧았습니다. Spark 시작 3.302초는 별도입니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/korean_v2/preprocessing.json

### 08. Hadoop 기반 저장·집계 검증

Hadoop은 원문을 뺀 출처·라벨·길이 데이터를 HDFS에 적재하고, MapReduce로 27개 키로 집계했습니다. 27개 모두 Python 독립 집계와 일치했고, 다시 읽은 Parquet 해시도 같았습니다. 단일 서버 실증입니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/eda_20261029/hadoop/verification.json, reports/eda_20261029/hadoop/job.log

### 09. HDFS–Spark 연동 파이프라인

피드백(속성 정리를 위한 것이 Spark·Hadoop 전처리)에 맞춰 두 도구를 HDFS로 연결했다. 원천을 HDFS에 올리고, Spark가 HDFS에서 읽어 정제한 결과를 다시 HDFS에 쓰며, MapReduce는 그 결과를 읽어 집계한다. 스크립트는 저장소 bigdata/에 있고, 실행 검증 결과는 재실행 후 반영한다.

> 근거 파일(/home/kyu/BigDB_Proj/): bigdata/README.md (worktree 브랜치 sangdelia/blobfish) · bigdata/run_pipeline.sh · bigdata/spark_pipeline.py · bigdata/verify_pipeline.py

### 10. 모델 구성과 입력·출력

모델은 두 개입니다. 모델 A는 6종 통합 데이터로 학습한 정상·유해 이진 분류기, 모델 B는 K-HATERS 원본 Ratings로 학습한 Hate·공격성·욕설 점수 모델입니다. 둘 다 문자 2~5gram TF-IDF와 LogisticRegression을 씁니다. 입력은 한글이 포함된 1,000자 이하 문장이고, 모델 A는 43점 이상이면 유해, 모델 B는 0~100 점수 세 개를 냅니다.

> 근거 파일(/home/kyu/BigDB_Proj/): src/bigdb_mvp/korean_model.py, reports/korean_short_v1/metrics.json, reports/khaters_score_v2/metrics.json

### 11. 모델 A 수식: 이진 로지스틱 회귀

피드백(모델 A·B를 만들었으면 수식)에 대한 답. 모델 A는 속성 x에 가중치 w를 곱해 더한 z를 시그모이드로 확률로 바꾼다. 학습은 L2 규제가 있는 가중 로지스틱 손실 최소화(scikit-learn liblinear)이고, 유해가 63%라 클래스 가중치로 균형을 맞췄다. 판정 기준 0.43은 validation에서 정했다.

> 근거 파일(/home/kyu/BigDB_Proj/): src/bigdb_mvp/experiments/classification.py · reports/korean_short_v1/char_2_5_200000.json

### 12. 모델 B 수식: 평점 3등급 분류 → 점수

모델 B는 K-HATERS의 항목별 평점(0·1·2)을 정답으로 3등급 분류를 학습하고, 등급 확률의 기댓값을 0–100점으로 바꾼다. Hate·공격성·욕설 각각 따로 학습한다.

> 근거 파일(/home/kyu/BigDB_Proj/): src/bigdb_mvp/experiments/khaters_score_train.py · src/bigdb_mvp/experiments/khaters_ratings.py · reports/khaters_score_v2/metrics.json

### 13. 모델 A (6종 이진 분류) test 결과

모델 A의 내부 test 결과는 Macro-F1 0.7784, 유해 재현율 83.52%, 정상 오탐률 27.82%입니다. 유해를 찾는 성능과 정상 문장의 오탐을 함께 봐야 합니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/korean_short_v1/metrics.json

### 14. 모델 A 혼동행렬과 지표 계산

피드백(27%가 어떻게 나왔나)에 대한 답. 31만 1,661행을 연결 그룹 단위로 나눈 test 31,349건에 모델 A를 적용한 혼동행렬이다. 실제 정상 11,599건 중 3,227건을 유해로 판정해 정상 오탐률 27.82%가 나온다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/korean_short_v1/char_2_5_200000.json (test tn/fp/fn/tp)

### 15. 얼마나 잘 만들었나: 기준선과 비교

피드백(얼마나 잘 만들었냐)에 대한 답. 모델 A는 모든 문장을 유해로 찍는 기준선(Macro-F1 0.3865)보다 크게 높고, 조각 길이·특징 수를 바꾼 설정 중 2–5gram·20만이 가장 좋았다. 모델 B는 항상 평균을 내는 기준선보다 세 점수 모두 오차가 낮다.

> 근거 파일(/home/kyu/BigDB_Proj/): [브랜치 sangdelia/blobfish] reports/postmidterm_20261105/majority.json · docs/progress/2026-10-10.md §16 · reports/khaters_score_v2/metrics.json

### 16. 데이터 규모 실험: 학습 데이터가 늘면 성능은?

피드백(추가 데이터로 테스트하면 정확도가 오를 것)에 대한 답. 학습 데이터 양만 바꿔 다시 학습해 보니 5%에서 100%로 늘릴수록 Macro-F1이 꾸준히 올랐다. 증가 폭은 줄지만 아직 포화되지 않아, 데이터를 더 모으면 개선 여지가 있다.

> 근거 파일(/home/kyu/BigDB_Proj/): [브랜치 sangdelia/blobfish] reports/postmidterm_20261105/learning_curve/ · [브랜치] experiments/learning_curve.py

### 17. 웹 실행과 부정문 오류

웹은 저장 모델을 API로 호출해 결과와 특징을 보여 줍니다. 실제 화면에서 일반 문장은 0.57로 정상이었지만, 부정문은 43.80으로 기준 43을 넘어 유해로 오탐했습니다. 오른쪽 QR이나 아래 주소로 직접 접속해 볼 수 있습니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/presentation_refresh_20261028/video/cue-02.png, cue-04.png

### 18. 오탐 원인과 판단 보류 규칙

발표 중 찾은 오탐(시래기, 부정문)을 고친 실험. 모델을 다시 학습하지 않고, 기준선 바로 위의 애매한 점수와 조각 하나가 점수를 좌우하는 짧은 입력을 '판단 보류'로 돌렸다. test에서 자동 판정 대상의 정상 오탐률이 4%p 넘게 줄었지만, 보류된 문장의 절반 이상은 실제 유해라 사람 검토가 필요하다. 처음 정한 선택 기준(재현율 유지)으로는 어떤 규칙도 뽑히지 않아 기준을 'Macro-F1 최대'로 바꾼 뒤 test를 1회 평가했다. 웹 서비스에는 아직 반영하지 않았다.

> 근거 파일(/home/kyu/BigDB_Proj/): [브랜치 sangdelia/blobfish] reports/postmidterm_20261105/fp_review3.json · [브랜치] experiments/fp_review3.py · artifacts/korean_short_v1/char_2_5_200000.joblib

### 19. 후속 실험 계획

후속 실험은 현재 모델을 기준선으로 고정하고 오류 유형 평가셋, 짧은 입력 보류, 문맥 모델, 출처×길이 분석을 비교합니다. 예를 들어 ‘시래기’는 74.33으로 유해 판정되는데, 학습 데이터 9건이 모두 ‘쓰레기’ 변형 욕설이었기 때문입니다.

> 근거 파일(/home/kyu/BigDB_Proj/): data/curated/korean_v2/corpus.parquet, artifacts/korean_short_v1/char_2_5_200000.joblib

### 20. 소논문 방향과 작성 일정

소논문 주제안은 다출처 한국어 유해표현 통합 데이터와 문자 기반 분류기의 오류 분석입니다. 11월 둘째 주까지 관련 연구와 평가셋, 셋째 주 비교 실험, 넷째 주 초안을 거쳐 11월 말에 마무리하고, 12월 초 최종 발표를 하겠습니다.

### 21. 프로젝트 R&R과 예상 산출물

역할은 조규상이 모델·실험, 서재철이 데이터·자료, 윤건호가 웹·서비스를 맡고, 결과 해석과 리허설은 공동으로 합니다. 예상 산출물은 정제 데이터, 모델 2종, 웹 시연, 소논문입니다. 감사합니다.

> 근거 파일(/home/kyu/BigDB_Proj/): docs/MIDTERM_PLAN_2026-11-04.md

## 부록

### A1. 부록 · 이진 모델의 출처별 test 성능

모델 A의 기존 내부 test를 출처별로 나눈 결과입니다. BEEP은 유해 재현율이 40.97%로 낮고, K-HATERS는 정상 오탐률이 38.78%로 높습니다. 출처마다 라벨 정의가 달라 출처 간 우열 비교는 아닙니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/korean_short_v1/metrics.json

### A2. 부록 · K-HATERS Ratings 3종 성능

모델 B의 세 점수 성능입니다. 공식 validation과 test 각 9,997건 기준으로 세 점수 모두 train 평균 예측보다 MAE가 낮습니다. 예를 들어 Hate는 32.44에서 19.77입니다. 점수는 정답 확률이나 피해 심각도가 아닙니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/khaters_score_v2/metrics.json

### A3. 부록 · 개선 후보와 선택

모델별 validation에서 개선 후보를 비교했습니다. Ratings 제한 정규화는 Hate MAE가 19.9905에서 19.8762로 낮아져 채택했고, 불균형 보정은 등급 0 오탐이 5.40%에서 21.78%로 늘어 채택하지 않았습니다. 이진 문자 2–8gram은 Macro-F1이 낮아 기존 모델을 유지했습니다. 이미 본 test의 후속 비교에서 욕설 MAE는 악화했습니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/midterm_20261104/comparison.csv

### A4. 부록 · 출처·길이별 오류 진단

고정 모델로 validation 오류를 다시 진단했습니다. BEEP 실제 유해의 50.77%를 놓쳤고, K-HATERS 실제 정상의 39.37%, 201자 이상 실제 정상의 56.65%를 오탐했습니다. 재학습이나 임계값 변경은 없었고, 원인은 후속으로 확인합니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/validation_audit_20261029/report.json

### A5. 부록 · 정규화·형태소 재학습 비교

소규모 표본으로 정규화와 형태소 특징을 재학습해 비교했습니다. 형태소를 넣으면 F1이 0.0102 올랐지만 재현율이 떨어지고 문맥 오류는 그대로여서 형태소는 제외했습니다. 소규모 표본·단일 seed 결과라 현재 운영 모델과 직접 비교하지 않습니다.

> 근거 파일(/home/kyu/BigDB_Proj/): docs/HYPOTHESIS_PILOT_RESULTS.md

### A6. 부록 · 선택 주석 미제공률과 필수화 비교

EDA에서 선택 주석 미제공률은 문맥 84.1%, 대상 43.6%, 표현 42.6%였습니다. 이를 모두 필수로 하면 7,149행, 전체의 2.3%만 남고 30만 4,512행을 제외하게 되므로 채택하지 않고 주석을 보존했습니다. Hadoop과 무관한 Python 분석입니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/eda_20261029/report.json

### A7. 부록 · 시연 화면 추가 사례

공개 웹 녹화 화면입니다. 욕설 문장은 95.57로 유해, 욕설 점수는 58.43입니다. 후보 구간을 삭제하면 98.78에서 87.02로 11.76점 내려갑니다. 특징이 없는 입력은 판단을 보류합니다. 표현 후보는 사람이 확정한 유해 구간이 아닙니다.

> 근거 파일(/home/kyu/BigDB_Proj/): reports/presentation_refresh_20261028/video/cue-03.png, cue-08.png, cue-09.png

### A8. 부록 · 시연 영상 (46초)

공개 웹에서 실제 입력 5개를 녹화한 46초 영상입니다. 자막 해설이 있고 소리는 없습니다. 공개 주소에 장애가 있으면 이 영상으로 대체합니다.

> 근거 파일(/home/kyu/BigDB_Proj/): ppts/midterm_20261104_r3/BigDB_시연_해설.mp4
