"""Download public SMS datasets into data/raw/."""
from pathlib import Path

from datasets import load_dataset

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HF_DATASETS = {
    "moz_smishing": "MOZNLP/MOZ-Smishing",
    "uci_sms_spam": "ucirvine/sms_spam",
    "electricsheep_africa_smishing": "electricsheepafrica/africa-smishing-sms-phishing",
}


def main():
    for name, repo in HF_DATASETS.items():
        try:
            ds = load_dataset(repo)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        for split, part in ds.items():
            out = RAW / f"{name}__{split}.csv"
            part.to_pandas().to_csv(out, index=False)
            print(f"[ok] {name}/{split}: {len(part)} rows -> {out.name}")


if __name__ == "__main__":
    main()
