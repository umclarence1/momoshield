"""One place that defines the train / test splits, so every model sees identical data."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scrub import scrub

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
PREDS = ROOT / "reports" / "preds"
SEED = 13


def load_splits():
    """Return (train_real, synth, tests) where tests maps name -> DataFrame with text, label."""
    df = pd.read_csv(PROC / "all.csv")
    pool = df[df.split_role == "pool"].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(pool, pool.label, groups=pool.campaign_id))
    train, test = pool.iloc[tr_idx], pool.iloc[te_idx]
    assert not set(train.campaign_id) & set(test.campaign_id)

    chal = pd.read_csv(ROOT / "data" / "challenge" / "ghana_challenge.csv")
    chal["text"] = chal["text"].map(scrub)

    tests = {
        "in_domain": test,
        "moz_dev": df[df.split_role == "moz_dev"],
        "moz_test": df[df.split_role == "moz_test"],
        "gh_challenge": chal,
    }
    return train, df[df.split_role == "synthetic"], tests


def save_preds(model, train_variant, test_name, frame, scores, threshold):
    PREDS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"text": frame["text"].values, "label": frame["label"].values,
                        "score": scores, "threshold": threshold})
    out.to_csv(PREDS / f"{model}__{train_variant}__{test_name}.csv", index=False)
