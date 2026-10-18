"""Compute metrics (with bootstrap 95% CIs) from every saved prediction file.

Writes reports/results.csv and reports/results.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "reports" / "preds"
N_BOOT = 1000
TEST_ORDER = ["in_domain", "moz_dev", "moz_test", "moz_test_pt", "gh_challenge"]


def point(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return rec, prec, fpr, f1, fn, fp


def ci(y, pred, rng):
    n = len(y)
    stats = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        stats.append(point(y[i], pred[i])[:3])
    lo, hi = np.nanpercentile(np.array(stats, dtype=float), [2.5, 97.5], axis=0)
    return lo, hi


def main():
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(PREDS.glob("*.csv")):
        model, train, test = f.stem.split("__")
        d = pd.read_csv(f)
        y = d.label.to_numpy()
        pred = (d.score.to_numpy() >= d.threshold.iloc[0]).astype(int)
        rec, prec, fpr, f1, fn, fp = point(y, pred)
        lo, hi = ci(y, pred, rng)
        pr_auc = average_precision_score(y, d.score) if 0 < y.sum() < len(y) else np.nan
        rows.append(dict(test=test, model=model, train=train, n=len(y), scams=int(y.sum()),
                         recall=rec, recall_lo=lo[0], recall_hi=hi[0],
                         precision=prec, fpr=fpr, fpr_lo=lo[2], fpr_hi=hi[2],
                         f1=f1, pr_auc=pr_auc, missed_scams=fn, false_alarms=fp))
    res = pd.DataFrame(rows)
    res["test"] = pd.Categorical(res.test, TEST_ORDER + sorted(set(res.test) - set(TEST_ORDER)))
    res = res.sort_values(["test", "model", "train"])
    res.to_csv(ROOT / "reports" / "results.csv", index=False)

    lines = ["| Test set | Model | Train | n (scams) | Recall [95% CI] | Precision | FPR [95% CI] | F1 | PR-AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        lines.append(f"| {r.test} | {r.model} | {r.train} | {r.n} ({r.scams}) | "
                     f"{r.recall:.2f} [{r.recall_lo:.2f}–{r.recall_hi:.2f}] | {r.precision:.2f} | "
                     f"{r.fpr:.3f} [{r.fpr_lo:.3f}–{r.fpr_hi:.3f}] | {r.f1:.2f} | {r.pr_auc:.2f} |")
    (ROOT / "reports" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
