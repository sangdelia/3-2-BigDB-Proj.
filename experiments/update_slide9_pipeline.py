"""PPT 9장(HDFS–Spark 연동 파이프라인)에 2026-11-05 실행 결과를 반영한다.
근거: $BIGDB_ROOT/reports/bigdata_pipeline/{spark.json, mapreduce.json, mapreduce.log, verification.json, hdfs-listing.txt}
usage: python update_slide9_pipeline.py DECK.pptx REPORT_DIR
"""
import json, re, sys
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor

deck, rep = Path(sys.argv[1]), Path(sys.argv[2])
spark = json.loads((rep / "spark.json").read_text())
ver = json.loads((rep / "verification.json").read_text())
mr = json.loads((rep / "mapreduce.json").read_text())
log = (rep / "mapreduce.log").read_text()
assert ver["status"] == "passed" and all(ver["checks"].values()), "verification did not pass"
map_out = int(re.findall(r"Map output records=(\d+)", log)[-1])
keys = ver["aggregate_keys"]
bronze_mb = int(re.search(r"(\d+) \S+ \S+ /bigdb/pipeline/bronze/raw_input.parquet", (rep / "hdfs-listing.txt").read_text())[1]) / 1e6

p = Presentation(str(deck))
s = p.slides[8]
assert s.shapes[8].text_frame.text.startswith("HDFS–Spark"), "slide 9 is not the pipeline slide"


def set_runs(shape_idx, para_idx, texts):
    runs = s.shapes[shape_idx].text_frame.paragraphs[para_idx].runs
    for r, t in zip(runs, texts):
        r.text = t
    for r in runs[len(texts):]:
        r.text = ""


ok = "✓ "
set_runs(12, 1, [f"원천 {spark['bronze_rows']:,}행 · {bronze_mb:.1f}MB"])
set_runs(12, 3, [ok + "HDFS 해시 = 로컬 해시"])
set_runs(14, 2, [f"정제 → {spark['silver_rows']:,}행 (−{spark['bronze_rows'] - spark['silver_rows']:,})"])
set_runs(14, 3, [ok + "출력 키 해시 = Python"])
set_runs(16, 2, [f"출처·라벨·길이 {keys}개 키"])
set_runs(16, 3, [ok + f"집계 {keys}/{keys} = Python"])
set_runs(18, 1, [f"Silver {ver['silver_rows']:,}행 입력"])
set_runs(18, 2, [f"{map_out:,}건 → {keys}개 키"])
set_runs(18, 3, [ok + "MapReduce = Spark"])
set_runs(19, 1, [f"실행 2026-11-05 · bigdata/run_pipeline.sh · Spark 작업 {spark['job_seconds']:.1f}초(시작 {spark['startup_seconds']:.1f}초 별도) · MapReduce {mr['job_seconds']}초"])
q = s.shapes[11].text_frame.paragraphs[0].runs
q[0].text = "실행 검증 통과 · 6개 검사 모두 일치"
q[0].font.color.rgb = RGBColor(0x3F, 0x7D, 0x62)
q[1].text = " · 단일 서버(Spark local[2], LocalJobRunner) · 다중 서버 확장성은 미측정"

tf = s.notes_slide.notes_text_frame
tf.text = (
    "발표 요점\n"
    "피드백(속성 정리를 위한 것이 Spark·Hadoop 전처리)에 맞춰 두 도구를 HDFS로 연결했다. 원천 316,571행을 HDFS Bronze에 올리고, "
    "Spark가 HDFS에서 읽어 정제한 311,661행을 Silver로 다시 HDFS에 쓰며, Spark SQL과 MapReduce가 각각 Silver를 읽어 27개 키로 집계한다. "
    f"2026-11-05 실행에서 HDFS 해시, 행 수, Spark 출력 키 해시, Spark 집계, MapReduce 집계가 모두 Python 기준과 일치했다(6/6). "
    "단일 서버 실행이라 속도·확장성 주장은 하지 않는다.\n\n"
    "근거 파일(/home/kyu/BigDB_Proj/): reports/bigdata_pipeline/verification.json · spark.json · mapreduce.log · hdfs-listing.txt · bigdata/run_pipeline.sh (worktree 브랜치 sangdelia/blobfish)"
)
p.save(str(deck))
print("slide 9 updated")
