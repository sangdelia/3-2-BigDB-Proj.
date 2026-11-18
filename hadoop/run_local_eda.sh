#!/usr/bin/env bash
# Run only the isolated BigDB HDFS namespace and a local Hadoop Streaming job.
set -euo pipefail

bigdb_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bigdb_runtime="${BIGDB_HADOOP_RUNTIME:?Set BIGDB_HADOOP_RUNTIME to the extracted Hadoop 3.4.2 directory}"
bigdb_state="$bigdb_root/data/processed/eda_20261029/hadoop_runtime"
bigdb_evidence="$bigdb_root/reports/eda_20261029/hadoop"
bigdb_python="$bigdb_root/.venv/bin/python"
mkdir -p "$bigdb_state"/{conf,logs,pids,nn,dn,tmp} "$bigdb_evidence"

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export HADOOP_HOME="$bigdb_runtime"
export HADOOP_CONF_DIR="$bigdb_state/conf"
export HADOOP_LOG_DIR="$bigdb_state/logs"
export HADOOP_PID_DIR="$bigdb_state/pids"
export HADOOP_IDENT_STRING=bigdb_eda_20261029
export HADOOP_HEAPSIZE_MAX=512
export HADOOP_OPTS="-XX:ActiveProcessorCount=2 -XX:+UseSerialGC -Xss512k"
cp "$bigdb_runtime/etc/hadoop/log4j.properties" "$HADOOP_CONF_DIR/log4j.properties"

cat > "$HADOOP_CONF_DIR/core-site.xml" <<EOF
<configuration>
  <property><name>fs.defaultFS</name><value>hdfs://127.0.0.1:19210</value></property>
  <property><name>hadoop.tmp.dir</name><value>$bigdb_state/tmp</value></property>
  <property><name>hadoop.http.max.threads</name><value>16</value></property>
  <property><name>ipc.server.read.threadpool.size</name><value>1</value></property>
</configuration>
EOF
cat > "$HADOOP_CONF_DIR/hdfs-site.xml" <<EOF
<configuration>
  <property><name>dfs.replication</name><value>1</value></property>
  <property><name>dfs.namenode.name.dir</name><value>file://$bigdb_state/nn</value></property>
  <property><name>dfs.datanode.data.dir</name><value>file://$bigdb_state/dn</value></property>
  <property><name>dfs.namenode.rpc-bind-host</name><value>127.0.0.1</value></property>
  <property><name>dfs.namenode.http-address</name><value>127.0.0.1:19870</value></property>
  <property><name>dfs.datanode.address</name><value>127.0.0.1:19866</value></property>
  <property><name>dfs.datanode.ipc.address</name><value>127.0.0.1:19867</value></property>
  <property><name>dfs.datanode.http.address</name><value>127.0.0.1:19864</value></property>
  <property><name>dfs.namenode.safemode.threshold-pct</name><value>0</value></property>
  <property><name>dfs.namenode.handler.count</name><value>3</value></property>
  <property><name>dfs.datanode.handler.count</name><value>2</value></property>
</configuration>
EOF
cat > "$HADOOP_CONF_DIR/mapred-site.xml" <<EOF
<configuration>
  <property><name>mapreduce.framework.name</name><value>local</value></property>
  <property><name>mapreduce.cluster.local.dir</name><value>$bigdb_state/tmp/mapred</value></property>
</configuration>
EOF

if [[ ! -f "$bigdb_state/nn/current/VERSION" ]]; then
  "$bigdb_runtime/bin/hdfs" namenode -format -nonInteractive > "$bigdb_evidence/format.log" 2>&1
fi
"$bigdb_runtime/bin/hdfs" --daemon start namenode
"$bigdb_runtime/bin/hdfs" --daemon start datanode
"$bigdb_runtime/bin/hdfs" dfsadmin -safemode wait > "$bigdb_evidence/readiness.txt" 2>&1

bigdb_namespace=/bigdb/eda_20261029
"$bigdb_runtime/bin/hdfs" dfs -mkdir -p "$bigdb_namespace/input" "$bigdb_namespace/curated"
"$bigdb_runtime/bin/hdfs" dfs -put -f \
  "$bigdb_root/data/curated/korean_v2/corpus.parquet" \
  "$bigdb_root/data/curated/korean_v2/manifest.json" "$bigdb_namespace/curated/"
"$bigdb_runtime/bin/hdfs" dfs -put -f \
  "$bigdb_root/data/processed/eda_20261029/hadoop_input.jsonl" "$bigdb_namespace/input/"

bigdb_output="$bigdb_namespace/results/run-$(date +%Y%m%d%H%M%S)-$$"
bigdb_started="$(date +%s)"
"$bigdb_runtime/bin/hadoop" jar \
  "$bigdb_runtime/share/hadoop/tools/lib/hadoop-streaming-3.4.2.jar" \
  -D mapreduce.framework.name=local \
  -D mapreduce.job.name=BigDB-Korean-EDA \
  -D mapreduce.job.reduces=1 \
  -D mapreduce.input.fileinputformat.split.maxsize=8388608 \
  -files "$bigdb_root/hadoop/mapper.py,$bigdb_root/hadoop/reducer.py" \
  -input "$bigdb_namespace/input/hadoop_input.jsonl" \
  -output "$bigdb_output" \
  -mapper "$bigdb_python mapper.py" \
  -combiner "$bigdb_python reducer.py" \
  -reducer "$bigdb_python reducer.py" > "$bigdb_evidence/job.log" 2>&1
bigdb_elapsed=$(( $(date +%s) - bigdb_started ))
"$bigdb_runtime/bin/hdfs" dfs -cat "$bigdb_output/part-00000" > "$bigdb_evidence/aggregates.tsv"
"$bigdb_runtime/bin/hdfs" dfs -cat "$bigdb_namespace/curated/corpus.parquet" \
  | sha256sum > "$bigdb_evidence/hdfs-corpus.sha256"
"$bigdb_runtime/bin/hdfs" dfs -count -h "$bigdb_namespace" > "$bigdb_evidence/hdfs-count.txt"
"$bigdb_runtime/bin/hdfs" dfsadmin -report > "$bigdb_evidence/hdfs-report.txt"
printf '{"job_output":"%s","job_seconds":%s,"execution_mode":"HDFS single node + MapReduce LocalJobRunner; YARN unused"}\n' \
  "$bigdb_output" "$bigdb_elapsed" > "$bigdb_evidence/run.json"
cd "$bigdb_root"
"$bigdb_python" -m bigdb_mvp.experiments.eda_hadoop_verify
