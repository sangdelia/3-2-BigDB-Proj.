"""Learning curve for model A: subsample train only; validation/test fixed; same recipe as char_2_5_200000."""
import json, random, sys, time
from pathlib import Path
from threadpoolctl import threadpool_limits
from bigdb_mvp.experiments.classification import run_method
from bigdb_mvp.experiments.classification_contracts import load_rows
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
rows, digest = load_rows(Path("data/curated/korean_v2"))
train = [r for r in rows if r.split == "train"]; rest = [r for r in rows if r.split != "train"]
rng = random.Random(42); order = list(range(len(train))); rng.shuffle(order)
# majority baseline on test (always predict harmful)
test = [r for r in rows if r.split == "test"]
pos = sum(1 for r in test if r.label.value == "harmful") if hasattr(test[0].label, "value") else sum(1 for r in test if r.label == "harmful")
n = len(test); f1h = 2 * pos / (pos + n); acc = pos / n
json.dump({"dataset_sha256": digest, "test_count": n, "test_harmful": pos, "majority_all_harmful": {"accuracy": acc, "harmful_f1": f1h, "clean_f1": 0.0, "macro_f1": f1h / 2, "harmful_recall": 1.0, "false_positive_rate": 1.0}}, open(OUT / "majority.json", "w"), indent=2)
print("majority", acc, f1h / 2, flush=True)
for frac in [float(x) for x in sys.argv[2:]]:
    k = round(len(train) * frac); sub = [train[i] for i in sorted(order[:k])]
    t = time.time(); (OUT / f"art_{frac}").mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        res = run_method("char_2_5_200000", tuple(sub + rest), OUT / f"art_{frac}", 200000, ngram_range=(2, 5))
    d = res.model_dump(); d["train_fraction"] = frac; d["wall_seconds"] = time.time() - t
    json.dump(d, open(OUT / f"lc_{frac}.json", "w"), indent=2, default=str)
    print(frac, k, "val", round(res.validation.macro_f1, 4), "test", round(res.test.macro_f1, 4), round(time.time() - t), "s", flush=True)
