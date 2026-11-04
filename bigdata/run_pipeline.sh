#!/usr/bin/env bash
# HDFS-integrated pipeline (single host):
#   local raw → HDFS Bronze → Spark (reads HDFS) → HDFS Silver/Gold → MapReduce (reads HDFS Silver) → HDFS Gold → cross-check
# Usage: BIGDB_ROOT=/home/kyu/BigDB_Proj BIGDB_HADOOP_RUNTIME=/path/hadoop-3.4.2 bash bigdata/run_pipeline.sh [--stop]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${BIGDB_ROOT:-/home/kyu/BigDB_Proj}"
runtime="${BIGDB_HADOOP_RUNTIME:-/tmp/bigdb-hadoop-20261029/runtime}"
py="$root/.venv/bin/python"
state="$root/data/processed/bigdata_pipeline/hadoop_runtime"
report="$root/reports/bigdata_pipeline"
ns=/bigdb/pipeline
fs=hdfs://127.0.0.1:19310
mkdir -p "$state"/{conf,logs,pids,nn,dn,tmp} "$report"

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export HADOOP_HOME="$runtime" HADOOP_CONF_DIR="$state/conf" HADOOP_LOG_DIR="$state/logs" HADOOP_PID_DIR="$state/pids"
export HADOOP_IDENT_STRING=bigdb_pipeline HADOOP_HEAPSIZE_MAX=512
export HADOOP_OPTS="-XX:ActiveProcessorCount=2 -XX:+UseSerialGC -XX:CICompilerCount=2 -Xss512k"
export POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
hdfs="$runtime/bin/hdfs"

if [[ "${1:-}" == "--stop" ]]; then
  "$hdfs" --daemon stop datanode || true; "$hdfs" --daemon stop namenode || true; exit 0
fi

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
  <property><name>dfs.namenode.safemode.threshold-pct</name><value>0</value></property>
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

echo "[1/6] HDFS start"
[[ -f "$state/nn/current/VERSION" ]] || "$hdfs" namenode -format -nonInteractive > "$report/format.log" 2>&1
"$hdfs" --daemon start namenode || true
"$hdfs" --daemon start datanode || true
"$hdfs" dfsadmin -safemode wait > "$report/readiness.txt" 2>&1

echo "[2/6] Bronze: local raw → HDFS"
"$hdfs" dfs -mkdir -p "$ns/bronze"
"$hdfs" dfs -put -f "$root/data/curated/korean_v2/raw_input.parquet" "$ns/bronze/"
"$hdfs" dfs -cat "$ns/bronze/raw_input.parquet" | sha256sum > "$report/hdfs-bronze.sha256"

echo "[3/6] Spark: HDFS Bronze → HDFS Silver/Gold"
cd "$root"
"$py" "$here/spark_pipeline.py" --fs "$fs" --ns "$ns" --report "$report" > "$report/spark.log" 2>&1 || { tail -30 "$report/spark.log"; exit 1; }
tail -1 "$report/spark.log"

echo "[4/6] MapReduce: HDFS Silver → HDFS Gold"
out="$ns/gold/mapreduce_aggregates"
"$hdfs" dfs -rm -r -f "$out" > /dev/null
started=$(date +%s)
"$runtime/bin/hadoop" jar "$runtime/share/hadoop/tools/lib/hadoop-streaming-3.4.2.jar" \
  -D mapreduce.framework.name=local -D mapreduce.job.name=BigDB-pipeline-gold -D mapreduce.job.reduces=1 \
  -files "$here/mapper.py,$here/reducer.py" \
  -input "$ns/silver/mr_input" -output "$out" \
  -mapper "$py mapper.py" -combiner "$py reducer.py" -reducer "$py reducer.py" > "$report/mapreduce.log" 2>&1
echo "{\"job_seconds\": $(( $(date +%s) - started ))}" > "$report/mapreduce.json"
"$hdfs" dfs -cat "$out/part-00000" > "$report/mapreduce_aggregates.tsv"

echo "[5/6] Verify: Python reference vs Spark vs MapReduce"
"$hdfs" dfs -ls -R "$ns" > "$report/hdfs-listing.txt"
"$py" "$here/verify_pipeline.py" --raw "$root/data/curated/korean_v2/raw_input.parquet" --report "$report"

echo "[6/6] done — daemons left running; stop with: bash $here/run_pipeline.sh --stop"
