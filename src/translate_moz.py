"""Machine-translate MOZ-Smishing (Portuguese) to English so English models can be tested on it."""
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "moz_smishing__test.csv"
OUT = ROOT / "data" / "raw" / "moz_smishing_en.csv"
MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"
BATCH = 32


def main():
    df = pd.read_csv(SRC)
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).eval()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    texts = df["text"].fillna("").astype(str).tolist()
    # Sort by length so each batch pads little; restore order afterwards.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [""] * len(texts)
    for b in range(0, len(order), BATCH):
        ids = order[b:b + BATCH]
        batch = tok([texts[i] for i in ids], return_tensors="pt", padding=True, truncation=True, max_length=160)
        limit = int(batch["input_ids"].shape[1] * 1.5) + 10
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=limit, num_beams=1, do_sample=False)
        for i, t in zip(ids, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t
        print(f"{min(b + BATCH, len(texts))}/{len(texts)}", flush=True)
    df["text_pt"] = df["text"]
    df["text"] = out
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
