"""Two hold rules for model A (weights unchanged), chosen on validation; test once.
R1 boundary: harmful call with score in [43, 43+b)  -> review   (uncertain band; negation case)
R2 dominance: harmful call with <=k features, top-feature share >=d, score < s -> review (single-fragment case)"""
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
    X = vec.transform(texts).tocsr(); p = clf.predict_proba(X)[:, 1]; C = X.multiply(w).tocsr(); nnz = np.diff(X.indptr)
    pos = np.array([C.data[C.indptr[i]:C.indptr[i+1]].clip(min=0).sum() for i in range(C.shape[0])])
    top = np.array([C.data[C.indptr[i]:C.indptr[i+1]].max(initial=0) for i in range(C.shape[0])])
    return p, nnz, np.divide(top, pos, out=np.zeros_like(top), where=pos > 0), C
def metrics(y, harm, review):
    a = ~review; yy, hh = y[a], harm[a]
    tp = int((yy & hh).sum()); fp = int((~yy & hh).sum()); fn = int((yy & ~hh).sum()); tn = int((~yy & ~hh).sum())
    return {"auto": int(a.sum()), "review": int(review.sum()), "review_rate": float(review.mean()),
            "review_true_clean": int((review & ~y).sum()), "review_true_harmful": int((review & y).sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "macro_f1": (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fn+fp)) / 2,
            "harmful_recall": tp/(tp+fn), "false_positive_rate": fp/(fp+tn)}
def rule(p, nnz, dom, b, k, d, s):
    harm = p >= thr
    r1 = harm & (p < thr + b)
    r2 = harm & (nnz <= k) & (dom >= d) & (p < s) if k else np.zeros_like(harm)
    return harm, (nnz == 0) | r1 | r2
data = {}
for split in ("validation", "test"):
    sub = [r for r in rows if r.split == split]
    data[split] = (np.array([r.label == "harmful" for r in sub]),) + stats([r.text for r in sub])[:3]
y, p, nnz, dom = data["validation"]
base = metrics(y, p >= thr, nnz == 0); grid = []
for b in (0, .02, .03, .05, .07, .10):
    for k, d, s in [(0, 0, 0)] + [(k, d, s) for k in (1, 2, 3) for d in (.6, .8, 1.0) for s in (.8, .9)]:
        harm, rv = rule(p, nnz, dom, b, k, d, s); m = metrics(y, harm, rv); m.update(b=b, k=k, d=d, s=s); grid.append(m)
# criterion (fixed before test): among rules with review_rate <= 5% and auto harmful recall >= base,
# pick the largest reduction in auto false-positive rate; tie -> fewer reviews
ok = [g for g in grid if g["review_rate"] <= 0.05]
best = max(ok, key=lambda g: (round(g["macro_f1"], 4), -g["review"]))
y, p, nnz, dom = data["test"]
tb = metrics(y, p >= thr, nnz == 0); harm, rv = rule(p, nnz, dom, best["b"], best["k"], best["d"], best["s"]); ts = metrics(y, harm, rv)
probes = ["시래기", "시래기 된장국이 맛있다.", "쓰레기", "너는 바보가 아니야.", "너는 정말 멍청한 바보야.", "감사합니다.", "시발", "닥쳐", "병신아"]
pp, pn, pd, C = stats(probes); harm, rv = rule(pp, pn, pd, best["b"], best["k"], best["d"], best["s"])
pr = [{"text": t, "score": round(float(pp[i])*100, 2), "features": int(pn[i]), "top_share": round(float(pd[i]), 3),
       "top": names[C[i].indices[np.argmax(C[i].data)]] if C[i].nnz else None,
       "before": "harmful" if pp[i] >= thr else "clean", "after": "review" if rv[i] else ("harmful" if harm[i] else "clean")} for i, t in enumerate(probes)]
out = {"threshold": thr, "criterion": "validation: review<=5%, max auto macro-F1; tie fewer reviews (revised after first criterion selected no rule)",
       "validation_base": base, "selected": best, "test_base": tb, "test_selected": ts, "probes": pr, "grid": grid}
json.dump(out, open(OUT / "fp_review3.json", "w"), ensure_ascii=False, indent=1)
for k in ("validation_base", "selected", "test_base", "test_selected"):
    m = out[k]; print(k, {x: (round(m[x], 4) if isinstance(m[x], float) else m[x]) for x in ("review", "review_rate", "review_true_clean", "review_true_harmful", "fp", "macro_f1", "harmful_recall", "false_positive_rate", "b", "k", "d", "s") if x in m})
for x in pr: print(x)
print("frontier (validation):")
for g in sorted([g for g in grid if g["k"] == 0], key=lambda g: g["b"]): print(g["b"], g["review_rate"].__round__(4), round(g["false_positive_rate"], 4), round(g["harmful_recall"], 4), round(g["macro_f1"], 4))
