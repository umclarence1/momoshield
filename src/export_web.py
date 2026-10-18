"""Export the TF-IDF + LR model for the in-browser site, plus a parity fixture.

    python src/export_web.py      -> web/model.json, web/parity_fixture.json
    node web/parity_test.js       -> checks the JS port matches Python
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules  # noqa: E402
from scrub import scrub  # noqa: E402
from splits import ROOT, load_splits  # noqa: E402
from text_model import TfidfLR  # noqa: E402,F401

WEB = ROOT / "web"


def vocab_table(vec, coefs):
    idf = vec.idf_
    return {term: [round(float(idf[j]), 6), round(float(coefs[j]), 6)] for term, j in vec.vocabulary_.items()}


def main():
    m = joblib.load(ROOT / "models" / "tfidf_lr_real_synth.joblib")
    coef = m.clf.coef_[0]
    nw = len(m.word.vocabulary_)
    model = {
        "intercept": round(float(m.clf.intercept_[0]), 6),
        "word": vocab_table(m.word, coef[:nw]),
        "char": vocab_table(m.char, coef[nw:]),
    }
    WEB.mkdir(exist_ok=True)
    (WEB / "model.json").write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Parity fixture: raw texts from every test set, with Python's outputs.
    _, _, tests = load_splits()
    chal = pd.read_csv(ROOT / "data" / "challenge" / "ghana_challenge.csv")
    raw = (chal.text.tolist()
           + tests["moz_test"].text_raw.tolist()
           + tests["in_domain"].text_raw.sample(400, random_state=0).tolist())
    clean = [scrub(t) for t in raw]
    probs = m.predict_proba(clean)
    fixture = [{"raw": r, "clean": c, "p": float(p), "rule_score": rules.check(c)[0]}
               for r, c, p in zip(raw, clean, probs)]
    (WEB / "parity_fixture.json").write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    size = (WEB / "model.json").stat().st_size / 1e6
    print(f"model.json {size:.2f} MB ({nw} word + {len(m.char.vocabulary_)} char features); fixture {len(fixture)} msgs")


if __name__ == "__main__":
    main()
