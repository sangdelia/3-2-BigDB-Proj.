#!/usr/bin/env bash
# HDFS Bronze -> Spark Silver/Gold -> MapReduce Gold -> Python verification.
# Usage: BIGDB_ROOT=/home/kyu/BigDB_Proj bash bigdata/run_pipeline.sh [--check|--stop]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${BIGDB_ROOT:-/home/kyu/BigDB_Proj}"
runtime="${BIGDB_HADOOP_RUNTIME:-/tmp/bigdb-hadoop-20261029/runtime}"
py="${BIGDB_PYTHON:-$root/.venv/bin/python}"
raw="${BIGDB_INPUT:-$root/data/curated/korean_v2/raw_input.parquet}"
run_id="${BIGDB_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export BIGDB_SPARK_WORKERS="${BIGDB_SPARK_WORKERS:-2}"
export BIGDB_PARTITIONS="${BIGDB_PARTITIONS:-4}"
maps="${BIGDB_MAP_TASKS:-2}"
reducers="${BIGDB_REDUCERS:-1}"
stage_timeout="${BIGDB_STAGE_TIMEOUT:-300}"
state="$root/data/processed/bigdata_pipeline/hadoop_runtime"
report="$root/reports/bigdata_pipeline/runs/$run_id"
ns="/bigdb/pipeline/runs/$run_id"
fs=hdfs://127.0.0.1:19310

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export HADOOP_HOME="$runtime" HADOOP_CONF_DIR="$state/conf"
export HADOOP_LOG_DIR="$state/logs" HADOOP_PID_DIR="$state/pids"
export HADOOP_IDENT_STRING=bigdb_pipeline HADOOP_HEAPSIZE_MAX=512
export HADOOP_OPTS="-XX:ActiveProcessorCount=2 -XX:+UseSerialGC -XX:CICompilerCount=2 -Xss512k"
export POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
hdfs="$runtime/bin/hdfs"

case "${1:-}" in
  --help)
    printf '%s\n' 'Usage: bash bigdata/run_pipeline.sh [--check|--stop]' \
      'BIGDB_ROOT, BIGDB_INPUT, BIGDB_RUN_ID select data and a new immutable run.' \
      'BIGDB_SPARK_WORKERS=2 BIGDB_PARTITIONS=4 BIGDB_MAP_TASKS=2 BIGDB_REDUCERS=1' \
      'Each stage has BIGDB_STAGE_TIMEOUT=300 seconds; pre-existing HDFS is reused.'
    exit 0 ;;
  --stop)
    "$hdfs" --daemon stop datanode
    "$hdfs" --daemon stop namenode
    exit 0 ;;
  ""|--check) ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
[[ "$#" -le 1 ]] || { printf 'Too many arguments\n' >&2; exit 2; }
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || {
  printf 'Invalid BIGDB_RUN_ID (letters, digits, dash, underscore; max 80)\n' >&2
  exit 2
}
for value in "$BIGDB_SPARK_WORKERS" "$maps" "$reducers"; do
  [[ "$value" =~ ^[1-8]$ ]] || { printf 'Worker counts must be 1..8\n' >&2; exit 2; }
done
[[ "$BIGDB_PARTITIONS" =~ ^[1-9][0-9]?$ ]] && (( BIGDB_PARTITIONS <= 64 )) || {
  printf 'BIGDB_PARTITIONS must be 1..64\n' >&2; exit 2
}
[[ "$stage_timeout" =~ ^[1-9][0-9]{0,3}$ ]] && (( stage_timeout <= 3600 )) || {
  printf 'BIGDB_STAGE_TIMEOUT must be 1..3600\n' >&2; exit 2
}
for file in "$py" "$JAVA_HOME/bin/java" "$hdfs" "$runtime/bin/hadoop"; do
  [[ -x "$file" ]] || { printf 'Missing executable: %s\n' "$file" >&2; exit 2; }
done
[[ -r "$raw" ]] || { printf 'Missing input: %s\n' "$raw" >&2; exit 2; }
[[ -r "$runtime/share/hadoop/tools/lib/hadoop-streaming-3.4.2.jar" ]] || {
  printf 'Hadoop Streaming 3.4.2 is required\n' >&2; exit 2
}
"$py" -c 'import pyspark, polars, pydantic, typer'
if [[ "${1:-}" == "--check" ]]; then
  printf 'preflight=passed\ninput=%s\nhdfs=%s\nspark_workers=%s\npartitions=%s\nmap_tasks=%s\nreducers=%s\n' \
    "$raw" "$fs" "$BIGDB_SPARK_WORKERS" "$BIGDB_PARTITIONS" "$maps" "$reducers"
  exit 0
fi

mkdir -p "$state"/{conf,logs,pids,nn,dn,tmp} "$root/reports/bigdata_pipeline/runs"
exec 9>"$state/pipeline.lock"
flock -n 9 || { printf 'Another pipeline run is active\n' >&2; exit 1; }
mkdir "$report" || { printf 'Run report already exists: %s\n' "$report" >&2; exit 1; }
stage=bootstrap
own_nn=0
own_dn=0
finish() {
  result=$?
  trap - EXIT
  if (( result != 0 )); then
    printf 'stage=%s\nexit_code=%s\n' "$stage" "$result" > "$report/failure.txt"
    printf 'Pipeline failed at %s; logs: %s\n' "$stage" "$report" >&2
  fi
  if (( own_dn )); then
    timeout 20 "$hdfs" --daemon stop datanode >> "$report/shutdown.log" 2>&1 || \
      printf 'DataNode shutdown needs inspection: %s\n' "$report/shutdown.log" >&2
  fi
  if (( own_nn )); then
    timeout 20 "$hdfs" --daemon stop namenode >> "$report/shutdown.log" 2>&1 || \
      printf 'NameNode shutdown needs inspection: %s\n' "$report/shutdown.log" >&2
  fi
  exit "$result"
}
trap finish EXIT
printf 'run_id=%s\ninput=%s\nnamespace=%s\nspark_workers=%s\npartitions=%s\nmap_tasks=%s\nreducers=%s\n' \
  "$run_id" "$raw" "$ns" "$BIGDB_SPARK_WORKERS" "$BIGDB_PARTITIONS" "$maps" "$reducers" \
  > "$report/settings.txt"

if [[ ! -f "$HADOOP_CONF_DIR/core-site.xml" ]]; then
  cp "$runtime/etc/hadoop/log4j.properties" "$HADOOP_CONF_DIR/"
  cat > "$HADOOP_CONF_DIR/core-site.xml" <<XML
<configuration>
  <property><name>fs.defaultFS</name><value>$fs</value></property>
  <property><name>hadoop.tmp.dir</name><value>$state/tmp</value></property>
  <property><name>hadoop.http.max.threads</name><value>16</value></property>
  <property><name>ipc.server.read.threadpool.size</name><value>1</value></property>
</configuration>
XML
  cat > "$HADOOP_CONF_DIR/hdfs-site.xml" <<XML
<configuration>
  <property><name>dfs.replication</name><value>1</value></property>
  <property><name>dfs.namenode.name.dir</name><value>file://$state/nn</value></property>
  <property><name>dfs.datanode.data.dir</name><value>file://$state/dn</value></property>
  <property><name>dfs.namenode.rpc-bind-host</name><value>127.0.0.1</value></property>
  <property><name>dfs.namenode.http-address</name><value>127.0.0.1:19970</value></property>
  <property><name>dfs.datanode.address</name><value>127.0.0.1:19966</value></property>
  <property><name>dfs.datanode.ipc.address</name><value>127.0.0.1:19967</value></property>
  <property><name>dfs.datanode.http.address</name><value>127.0.0.1:19964</value></property>
  <property><name>dfs.namenode.handler.count</name><value>3</value></property>
  <property><name>dfs.datanode.handler.count</name><value>2</value></property>
</configuration>
XML
  cat > "$HADOOP_CONF_DIR/mapred-site.xml" <<XML
<configuration>
  <property><name>mapreduce.framework.name</name><value>local</value></property>
  <property><name>mapreduce.cluster.local.dir</name><value>$state/tmp/mapred</value></property>
</configuration>
XML
fi

printf '[1/6] HDFS readiness\n'
if ! timeout 15 "$hdfs" dfs -test -d / > "$report/probe.log" 2>&1; then
  [[ -f "$state/nn/current/VERSION" ]] || \
    timeout 45 "$hdfs" namenode -format -nonInteractive > "$report/format.log" 2>&1
  "$hdfs" --daemon start namenode > "$report/start.log" 2>&1
  own_nn=1
  "$hdfs" --daemon start datanode >> "$report/start.log" 2>&1
  own_dn=1
fi
timeout 45 "$hdfs" dfsadmin -safemode wait > "$report/readiness.txt" 2>&1
"$hdfs" dfs -mkdir -p /bigdb/pipeline/runs
"$hdfs" dfs -mkdir "$ns"

stage=bronze
printf '[2/6] Bronze: source input -> HDFS (%s)\n' "$run_id"
"$hdfs" dfs -mkdir "$ns/bronze"
timeout "$stage_timeout" "$hdfs" dfs -put "$raw" "$ns/bronze/raw_input.parquet"
timeout "$stage_timeout" "$hdfs" dfs -cat "$ns/bronze/raw_input.parquet" \
  | sha256sum > "$report/hdfs-bronze.sha256"

stage=spark
printf '[3/6] Spark: HDFS Bronze -> Silver/Gold\n'
cd "$here/.."
timeout --kill-after=10 "$stage_timeout" "$py" -m bigdata.spark_pipeline \
  --fs "$fs" --ns "$ns" --report "$report" > "$report/spark.log" 2>&1
tail -1 "$report/spark.log"

stage=mapreduce
printf '[4/6] MapReduce: HDFS Silver -> Gold (max %s maps at once)\n' "$maps"
started=$(date +%s)
timeout --kill-after=10 "$stage_timeout" "$runtime/bin/hadoop" \
  jar "$runtime/share/hadoop/tools/lib/hadoop-streaming-3.4.2.jar" \
  -D mapreduce.framework.name=local -D mapreduce.job.name=BigDB-pipeline-gold \
  -D "mapreduce.local.map.tasks.maximum=$maps" \
  -D "mapreduce.local.reduce.tasks.maximum=$reducers" -D "mapreduce.job.reduces=$reducers" \
  -files "$here/mapper.py,$here/reducer.py" \
  -input "$ns/silver/mr_input" -output "$ns/gold/mapreduce_aggregates" \
  -mapper "$py mapper.py" -combiner "$py reducer.py" -reducer "$py reducer.py" \
  > "$report/mapreduce.log" 2>&1
printf '{"job_seconds": %s, "max_parallel_maps": %s, "reducers": %s}\n' \
  "$(( $(date +%s) - started ))" "$maps" "$reducers" > "$report/mapreduce.json"
"$hdfs" dfs -cat "$ns/gold/mapreduce_aggregates/part-*" > "$report/mapreduce_aggregates.tsv"

stage=verification
printf '[5/6] Verify: Python reference vs stored Spark/MapReduce results\n'
"$hdfs" dfs -ls -R "$ns" > "$report/hdfs-listing.txt"
timeout "$stage_timeout" "$py" -m bigdata.verify_pipeline --raw "$raw" --report "$report"
"$hdfs" dfs -put "$report/verification.json" "$ns/verification.json"
printf 'elapsed_seconds=%s\n' "$SECONDS" > "$report/elapsed.txt"
printf '[6/6] verified\nreport=%s\nhdfs=%s%s\n' "$report" "$fs" "$ns"
