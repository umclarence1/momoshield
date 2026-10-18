"""Combine raw datasets into one scrubbed, de-duplicated, campaign-grouped table.

Output: data/processed/all.csv with columns
  text, text_raw, label (1 = scam/spam/smishing), orig_label, source, lang,
  is_synthetic, campaign_id, split_role
split_role:
  "pool"      - English real data (UCI + Mendeley): used for train / in-domain test
  "synthetic" - synthetic Ghana data: training only
  "moz_test"  - translated MOZ-Smishing: cross-country test only
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from scrub import scrub

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
NEAR_DUP_SIM = 0.8


def load_uci():
    d = pd.read_csv(RAW / "uci_sms_spam__train.csv")
    return pd.DataFrame({"text_raw": d["sms"], "orig_label": d["label"].map({0: "ham", 1: "spam"}),
                         "source": "uci", "lang": "en", "is_synthetic": 0})


def load_mendeley():
    d = pd.read_csv(RAW / "mendeley" / "Dataset_5971.csv", encoding="latin-1")
    return pd.DataFrame({"text_raw": d["TEXT"], "orig_label": d["LABEL"].str.lower(),
                         "source": "mendeley", "lang": "en", "is_synthetic": 0})


def load_moz():
    d = pd.read_csv(RAW / "moz_smishing_en.csv")
    return pd.DataFrame({"text_raw": d["text"], "text_pt_raw": d["text_pt"], "orig_label": d["label"].str.lower(),
                         "source": "moz_" + d["source"], "lang": "pt->en", "is_synthetic": 0})


def load_synth():
    d = pd.read_csv(PROC / "synthetic_ghana.csv")
    return pd.DataFrame({"text_raw": d["text"], "orig_label": np.where(d["label"] == 1, "scam", "genuine"),
                         "source": d["source"], "lang": "en", "is_synthetic": 1, "campaign_id": d["campaign_id"]})


def norm_key(s):
    return " ".join(str(s).lower().split())


def group_near_duplicates(texts, prefix):
    """Connected components of messages with TF-IDF cosine similarity >= NEAR_DUP_SIM."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(texts)
    nn = NearestNeighbors(radius=1 - NEAR_DUP_SIM, metric="cosine").fit(vec)
    _, idx = nn.radius_neighbors(vec)
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, neigh in enumerate(idx):
        for j in neigh:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    return [f"{prefix}{find(i)}" for i in range(len(texts))]


def main():
    pool = pd.concat([load_uci(), load_mendeley()], ignore_index=True)
    pool["split_role"] = "pool"
    moz = load_moz()
    moz["split_role"] = "moz_test"
    synth = load_synth()
    synth["split_role"] = "synthetic"

    df = pd.concat([pool, moz, synth], ignore_index=True)
    df["text_raw"] = df["text_raw"].fillna("").astype(str).str.strip()
    df = df[df["text_raw"].str.len() > 0]
    df["label"] = df["orig_label"].isin(["spam", "smishing", "scam"]).astype(int)
    df["text"] = df["text_raw"].map(scrub)
    # Original Portuguese for MOZ rows, so a multilingual model can be tested without translation.
    df["text_pt"] = df["text_pt_raw"].map(lambda s: scrub(s) if isinstance(s, str) else "")
    df = df.drop(columns="text_pt_raw")

    # Exact duplicates (after scrubbing) — Mendeley re-uses many UCI messages.
    before = len(df)
    df["_key"] = df["text"].map(norm_key)
    conflicts = df.groupby(["split_role", "_key"])["label"].nunique()
    n_conflict = int((conflicts > 1).sum())
    # Synthetic rows are kept even when scrubbing makes them identical: the copies act as
    # sample weights, otherwise ~1,400 generated rows collapse to ~200 and barely register.
    dup = df.duplicated(subset=["split_role", "_key"], keep="first") & (df["split_role"] != "synthetic")
    df = df[~dup]
    print(f"exact-duplicate rows removed: {before - len(df)} (label conflicts among them: {n_conflict})")

    # Near-duplicate campaign groups for real data; synthetic already has template ids.
    for role in ["pool", "moz_test"]:
        m = df["split_role"] == role
        df.loc[m, "campaign_id"] = group_near_duplicates(df.loc[m, "text"].tolist(), f"{role}_")

    # A message that also appears (after scrubbing) in the MOZ test set must not be trained on.
    moz_keys = set(df.loc[df.split_role == "moz_test", "_key"])
    leak = df["split_role"].isin(["pool", "synthetic"]) & df["_key"].isin(moz_keys)
    df = df[~leak]
    print(f"train-side rows dropped for overlapping the MOZ test set: {int(leak.sum())}")

    # MOZ dev/test split by campaign group: dev for error analysis and decisions, test reported.
    moz_groups = sorted(df.loc[df.split_role == "moz_test", "campaign_id"].unique())
    rng = np.random.default_rng(7)
    dev_groups = set(rng.choice(moz_groups, size=len(moz_groups) // 2, replace=False))
    df.loc[(df.split_role == "moz_test") & df.campaign_id.isin(dev_groups), "split_role"] = "moz_dev"

    df = df.drop(columns="_key")
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "all.csv"
    df.to_csv(out, index=False)
    summary = df.groupby(["split_role", "source", "label"]).size().unstack(fill_value=0)
    print(summary)
    for role in ["pool", "moz_test"]:
        g = df[df.split_role == role].groupby("campaign_id").size()
        print(f"{role}: {len(g)} campaign groups, largest = {g.max()}, groups with >1 msg = {(g > 1).sum()}")
    print("->", out)


if __name__ == "__main__":
    main()
