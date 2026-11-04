# HDFS–Spark 연동 파이프라인

중간발표에서는 Spark(정제 비교)와 Hadoop(저장·집계)을 **따로** 실행했다. 이 폴더는 두 도구를 HDFS 위에서 하나의 흐름으로 잇는다.

```
로컬 원천 행(316,571)
  └─[HDFS put]→ Bronze  /bigdb/pipeline/bronze/raw_input.parquet
        └─[Spark: HDFS에서 읽기 → 공백 정규화·충돌 제외·중복 병합]
              → Silver  /bigdb/pipeline/silver/corpus/        (정제 행, 길이 포함)
              → Silver  /bigdb/pipeline/silver/mr_input/      (출처·라벨·길이만, 원문 제외)
              → Gold    /bigdb/pipeline/gold/spark_aggregates/ (Spark 집계 27개 키)
        └─[MapReduce: HDFS Silver 읽기]
              → Gold    /bigdb/pipeline/gold/mapreduce_aggregates/ (MapReduce 집계)
  └─[검증] Python 기준 결과 = Spark 결과 = MapReduce 결과
```

| 단계 | 도구 | 입력 → 출력 | 검증 |
|---|---|---|---|
| Bronze 적재 | HDFS | 로컬 Parquet → HDFS | HDFS에서 다시 읽은 SHA-256 = 로컬 SHA-256 |
| Silver 정제 | Spark (`spark_pipeline.py`) | HDFS Bronze → HDFS Silver | 출력 키 해시 = Python 순차 처리 해시 |
| Gold 집계 | Spark SQL, MapReduce (`mapper.py`, `reducer.py`) | HDFS Silver → HDFS Gold | 27개 키 Spark = MapReduce = Python |

## 실행

```bash
cd /home/kyu/orca/workspaces/BigDB_Proj/blobfish
BIGDB_ROOT=/home/kyu/BigDB_Proj BIGDB_HADOOP_RUNTIME=/tmp/bigdb-hadoop-20261029/runtime bash bigdata/run_pipeline.sh
bash bigdata/run_pipeline.sh --stop   # 이 파이프라인의 HDFS 데몬만 종료
```

- 결과: `$BIGDB_ROOT/reports/bigdata_pipeline/` (`spark.json`, `mapreduce_aggregates.tsv`, `verification.json`, `hdfs-listing.txt`, 로그)
- HDFS 상태: `$BIGDB_ROOT/data/processed/bigdata_pipeline/hadoop_runtime/` (포트 19310·199xx, loopback 전용, 기존 EDA namespace와 분리)
- 필요: Java 11, Hadoop 3.4.2 배포본, 프로젝트 `.venv`(pyspark 3.5.7, polars, pydantic)

## 범위와 한계

- 단일 서버: HDFS 복제 수 1, Spark `local[2]`, MapReduce LocalJobRunner, YARN 미사용. 다중 서버 확장성은 측정하지 않는다.
- Spark가 맡는 것은 정제(정규화·충돌 제외·중복 병합)와 집계다. 연결 그룹 분할과 모델 학습은 기존 Python 단계가 그대로 한다.
- Silver는 중간발표 Spark 비교와 같은 규칙(출력 키 해시 `d10cf2e8…`)이다. 최종 코퍼스(`corpus.parquet`)와 행 수 311,661·라벨 분포는 같지만, 여러 출처에 걸친 중복 문장의 대표 출처(여기서는 사전순 최솟값)와 길이 기준(여기서는 정규화 문장)이 달라 출처별·길이별 집계가 수십 건씩 다르다.
- 스레드 수가 제한된 환경(예: 에이전트 샌드박스)에서는 JVM이 스레드를 만들지 못해 HDFS·Spark가 시작되지 않는다. 일반 터미널에서 실행한다.
