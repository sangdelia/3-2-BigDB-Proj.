"""False-positive mitigation for model A: hold (review) harmful calls dominated by very few features.
Rule selected on validation only; test evaluated once. Model weights unchanged."""
import json, sys
from pathlib import Path
import joblib, numpy as np
from bigdb_mvp.experiments.classification_contracts import load_rows
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
model = joblib.load("artifacts/korean_short_v1/char_2_5_200000.joblib")
vec, clf, thr = model.vectorizer, model.classifier, model.threshold
w = clf.coef_[0]; names = vec.get_feature_names_out()
rows, _ = load_rows(Path("data/curated/korean_v2"))
def stats(texts):
    X = vec.transform(texts).tocsr()
    p = clf.predict_proba(X)[:, 1]
    C = X.multiply(w).tocsr()
    nnz = np.diff(X.indptr)
    pos = np.array([C.data[C.indptr[i]:C.indptr[i+1]].clip(min=0).sum() for i in range(C.shape[0])])
    top = np.array([C.data[C.indptr[i]:C.indptr[i+1]].max(initial=0) for i in range(C.shape[0])])
    dom = np.divide(top, pos, out=np.zeros_like(top), where=pos > 0)
    return p, nnz, dom, C
def metrics(y, harm, review):
    a = ~review; yy, hh = y[a], harm[a]
    tp = int((yy & hh).sum()); fp = int((~yy & hh).sum()); fn = int((yy & ~hh).sum()); tn = int((~yy & ~hh).sum())
    f1h = 2*tp/(2*tp+fp+fn); f1c = 2*tn/(2*tn+fn+fp)
    return {"auto": int(a.sum()), "review": int(review.sum()), "review_rate": float(review.mean()),
            "review_true_clean": int((review & ~y).sum()), "review_true_harmful": int((review & y).sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "macro_f1": (f1h+f1c)/2,
            "harmful_recall": tp/(tp+fn), "false_positive_rate": fp/(fp+tn)}
res = {}
for split in ("validation", "test"):
    sub = [r for r in rows if r.split == split]
    y = np.array([r.label == "harmful" for r in sub]); p, nnz, dom, _ = stats([r.text for r in sub])
    res[split] = (y, p >= thr, nnz, dom)
y, harm, nnz, dom = res["validation"]
base = metrics(y, harm, (nnz == 0))
grid = []
for k in (3, 5, 8, 12, 16, 20, 30):
    for d in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        rv = (nnz == 0) | (harm & (nnz <= k) & (dom >= d))
        m = metrics(y, harm, rv); m.update(k=k, d=d); grid.append(m)
# pre-declared criterion: review rate <= 1%, auto recall drop <= 0.5%p, maximize FPR reduction; tie -> fewer reviews
ok = [g for g in grid if g["review_rate"] <= 0.01 and base["harmful_recall"] - g["harmful_recall"] <= 0.005]
best = min(ok, key=lambda g: (g["false_positive_rate"], g["review"])) if ok else None
out = {"threshold": thr, "criterion": "validation: review<=1%, recall drop<=0.5%p, min FPR", "validation_base": base, "selected": best, "grid": grid}
if best:
    y, harm, nnz, dom = res["test"]
    out["test_base"] = metrics(y, harm, nnz == 0)
    out["test_selected"] = metrics(y, harm, (nnz == 0) | (harm & (nnz <= best["k"]) & (dom >= best["d"])))
probes = ["시래기", "시래기 된장국이 맛있다.", "쓰레기", "너는 바보가 아니야.", "너는 정말 멍청한 바보야.", "감사합니다.", "시발", "닥쳐"]
p, nnz, dom, C = stats(probes); pr = []
for i, t in enumerate(probes):
    row = C[i]; idx = row.indices[np.argsort(-row.data)][:3]
    label = "review" if nnz[i] == 0 or (best and p[i] >= thr and nnz[i] <= best["k"] and dom[i] >= best["d"]) else ("harmful" if p[i] >= thr else "clean")
    pr.append({"text": t, "score": round(float(p[i]) * 100, 2), "features": int(nnz[i]), "top_share": round(float(dom[i]), 3),
               "top": [(names[j], round(float(C[i, j]), 2)) for j in idx], "before": "harmful" if p[i] >= thr else "clean", "after": label})
out["probes"] = pr
json.dump(out, open(OUT / "fp_review.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("validation_base", "selected")}, ensure_ascii=False, indent=0))
print(json.dumps({k: out.get(k) for k in ("test_base", "test_selected")}, ensure_ascii=False))
for x in pr: print(x)
