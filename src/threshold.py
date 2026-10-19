"""Choose decision thresholds on MOZ dev, report them on MOZ test.

The default 0.5 threshold flags about 28% of genuine mobile-money SMS. This picks the threshold
that meets a target false-positive rate on the dev half and shows what it costs in recall on the
test half, so the trade-off is a stated choice. Output: reports/threshold_analysis.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "reports" / "preds"
MODEL = "tfidf_lr__real+synth+spelling"
TARGET_FPR = [0.05, 0.10, 0.20]


def rates(y, s, thr):
    pred = s >= thr
    rec = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
    fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
    prec = (pred & (y == 1)).sum() / max(1, pred.sum())
    return rec, fpr, prec


def threshold_for_fpr(y, s, target):
    """Lowest threshold whose false-positive rate on (y, s) is at most `target`."""
    neg = np.sort(s[y == 0])[::-1]
    k = int(np.floor(target * len(neg)))
    return float(neg[k]) + 1e-9 if k < len(neg) else 0.0


def main():
    dev = pd.read_csv(PREDS / f"{MODEL}__moz_dev.csv")
    test = pd.read_csv(PREDS / f"{MODEL}__moz_test.csv")
    ind = pd.read_csv(PREDS / f"{MODEL}__in_domain.csv")
    rows = [("default", 0.5)] + [(f"FPR ≤ {int(t * 100)}% on dev", threshold_for_fpr(dev.label.values, dev.score.values, t))
                                   for t in TARGET_FPR]
    lines = ["| Threshold rule | Threshold | MOZ test recall | MOZ test FPR | MOZ test precision | English test recall | English test FPR |",
             "|---|---|---|---|---|---|---|"]
    for name, thr in rows:
        r, f, p = rates(test.label.values, test.score.values, thr)
        ri, fi, _ = rates(ind.label.values, ind.score.values, thr)
        lines.append(f"| {name} | {thr:.3f} | {r:.2f} | {f:.2f} | {p:.2f} | {ri:.2f} | {fi:.3f} |")
    out = "\n".join(lines) + "\n"
    (ROOT / "reports" / "threshold_analysis.md").write_text(out, encoding="utf-8")
    print(out.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
