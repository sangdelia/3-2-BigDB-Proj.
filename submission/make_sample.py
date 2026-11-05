"""Build the ~1 MB input-data sample for submission (run from /home/kyu/BigDB_Proj).

Only sources whose licenses allow redistribution with attribution are sampled:
K-HATERS (CC-BY-4.0), K-MHaS (CC-BY-SA-4.0), BEEP (CC-BY-SA-4.0).
NIKL / UnSmile / KOLD are excluded (redistribution not permitted or not declared).
Stratified by source x label, proportional to the corpus, seed 20261105.

Model B targets: K-HATERS rows also carry the original-Ratings grades used by model B
(b_hate = max of 7 group ratings, b_abuse = max of all 13, b_swear = swear; 0/1/2) and the
row's split in model B's cleaned dataset. K-HATERS rows absent from B's cleaned data
(19 of 168,727) are not sampled. Other sources leave these columns empty.
"""
import csv, io, sys
from pathlib import Path
import polars as pl

sys.path.insert(0, "src")
from bigdb_mvp.experiments.khaters_ratings import KINDS, load_dataset, text_key

ds = load_dataset(Path("data/raw/khaters_ratings/fe022af988c3dc5240340fc89629ca1e153fe8bd"))
B = {text_key(r.text): ([r.grade(k) for k in KINDS], name)
     for name, rows in zip(("train", "validation", "test"), ds.splits) for r in rows}

OUT = Path(sys.argv[1]); LIMIT = 1_000_000
SOURCES = ["khaters", "kmhas", "beep"]
df = (pl.read_parquet("data/curated/korean_v2/corpus.parquet", columns=["text", "label", "source", "split"])
        .with_columns(pl.col("text").map_elements(text_key, return_dtype=pl.String).alias("key"))
        .filter(pl.col("source").is_in(SOURCES))
        .filter((pl.col("source") != "khaters") | pl.col("text").map_elements(lambda t: text_key(t) in B, return_dtype=pl.Boolean))
        .with_columns(pl.col("text").str.replace_all(r"[\r\n\t]+", " ").str.strip_chars()))
shuffled = df.sample(fraction=1.0, shuffle=True, seed=20261105)
groups = {k: g for k, g in shuffled.group_by(["source", "label"], maintain_order=True)}
share = {k: g.height / df.height for k, g in groups.items()}

def render(n):
    buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
    w.writerow(["sample_id", "text", "label", "label_id", "source", "split", "length",
                "b_hate", "b_abuse", "b_swear", "b_split"])
    rows = []
    for (src, lab), g in sorted(groups.items()):
        rows += [(r["text"], r["key"], lab, src, r["split"]) for r in g.head(max(1, round(n * share[(src, lab)]))).iter_rows(named=True)]
    for i, (t, key, lab, src, sp) in enumerate(rows, 1):
        grades, bsplit = B[key] if src == "khaters" else (["", "", ""], "")
        w.writerow([f"s{i:05d}", t, lab, int(lab == "harmful"), src, sp, len(t), *grades, bsplit])
    return buf.getvalue(), len(rows)

lo, hi = 100, 20000
while lo < hi:                      # largest n whose UTF-8 CSV (with BOM) fits in 1 MB
    mid = (lo + hi + 1) // 2
    if len(("﻿" + render(mid)[0]).encode("utf-8")) <= LIMIT: lo = mid
    else: hi = mid - 1
text, n = render(lo)
OUT.write_text("﻿" + text, encoding="utf-8")   # BOM so Excel opens Korean correctly
print(n, "rows", OUT.stat().st_size, "bytes")
