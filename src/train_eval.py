"""Train the baselines and save their predictions on every test set.

Models
  rules        - hand-written red flags (src/rules.py), no training
  tfidf_lr     - TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression
Training data variants
  real         - English real data (UCI + Mendeley), train split only
  real+synth   - the above plus synthetic Ghana messages
Test sets (see splits.py)
  in_domain    - held-out English real data, split by campaign group
  moz_dev      - half of the real Mozambican mobile-money SMS (machine-translated), for error analysis
  moz_test     - the other half, by campaign group; the number to report
  gh_challenge - hand-written Ghana challenge set (constructed, small)
Metrics are computed by summarize.py from the saved predictions.
"""
import json

import joblib
import pandas as pd

import rules
from splits import ROOT, load_splits, save_preds
from text_model import TfidfLR

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def main():
    train, synth, tests = load_splits()

    for tname, t in tests.items():
        save_preds("rules", "-", tname, t, rules.predict_score(t.text.tolist()), rules.THRESHOLD)

    models = {}
    for variant, tr in [("real", train), ("real+synth", pd.concat([train, synth]))]:
        m = TfidfLR().fit(tr.text.tolist(), tr.label.values)
        models[variant] = m
        for tname, t in tests.items():
            save_preds("tfidf_lr", variant, tname, t, m.predict_proba(t.text.tolist()), 0.5)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(models["real+synth"], MODELS / "tfidf_lr_real_synth.joblib")
    joblib.dump(models["real"], MODELS / "tfidf_lr_real.joblib")

    # Error analysis on the MOZ *dev* half only; the test half stays unseen.
    m = models["real+synth"]
    moz = tests["moz_dev"].assign(p=m.predict_proba(tests["moz_dev"].text.tolist()))
    chal = tests["gh_challenge"]
    chal = chal.assign(p=m.predict_proba(chal.text.tolist()), rule_score=rules.predict_score(chal.text.tolist()))
    err = {
        "moz_dev_missed_scams": moz[(moz.label == 1)].nsmallest(15, "p")[["text", "p"]].to_dict("records"),
        "moz_dev_false_alarms": moz[(moz.label == 0)].nlargest(15, "p")[["text", "p"]].to_dict("records"),
        "challenge_errors": chal[(chal.p >= 0.5) != (chal.label == 1)][["kind", "text", "p", "rule_score"]].to_dict("records"),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "error_analysis.json").write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved reports/preds/*, reports/error_analysis.json, models/*.joblib")


if __name__ == "__main__":
    main()
