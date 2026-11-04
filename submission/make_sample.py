"""Build the ~1 MB input-data sample for submission (run from /home/kyu/BigDB_Proj).

Only sources whose licenses allow redistribution with attribution are sampled:
K-HATERS (CC-BY-4.0), K-MHaS (CC-BY-SA-4.0), BEEP (CC-BY-SA-4.0).
NIKL / UnSmile / KOLD are excluded (redistribution not permitted or not declared).
Stratified by source x label, proportional to the corpus, seed 20261105.
"""
import csv, io, sys
from pathlib import Path
import polars as pl

OUT = Path(sys.argv[1]); LIMIT = 1_000_000
SOURCES = ["khaters", "kmhas", "beep"]
df = (pl.read_parquet("data/curated/korean_v2/corpus.parquet", columns=["text", "label", "source", "split"])
        .filter(pl.col("source").is_in(SOURCES))
        .with_columns(pl.col("text").str.replace_all(r"[\r\n\t]+", " ").str.strip_chars()))
shuffled = df.sample(fraction=1.0, shuffle=True, seed=20261105)
groups = {k: g for k, g in shuffled.group_by(["source", "label"], maintain_order=True)}
share = {k: g.height / df.height for k, g in groups.items()}

def render(n):
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["sample_id", "text", "label", "label_id", "source", "split", "length"])
    rows = []
    for (src, lab), g in sorted(groups.items()):
        rows += [(r["text"], lab, src, r["split"]) for r in g.head(max(1, round(n * share[(src, lab)]))).iter_rows(named=True)]
    for i, (t, lab, src, sp) in enumerate(rows, 1):
        w.writerow([f"s{i:05d}", t, lab, int(lab == "harmful"), src, sp, len(t)])
    return buf.getvalue(), len(rows)

lo, hi = 100, 20000
while lo < hi:                      # largest n whose UTF-8 CSV (with BOM) fits in 1 MB
    mid = (lo + hi + 1) // 2
    if len(("﻿" + render(mid)[0]).encode("utf-8")) <= LIMIT: lo = mid
    else: hi = mid - 1
text, n = render(lo)
OUT.write_text("﻿" + text, encoding="utf-8")   # BOM so Excel opens Korean correctly
print(n, "rows", OUT.stat().st_size, "bytes")
