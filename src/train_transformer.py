"""Fine-tune a transformer on the same splits as the baselines and save its predictions.

    python src/train_transformer.py --model distilbert-base-uncased
    python src/train_transformer.py --model distilbert-base-multilingual-cased --tag mdistilbert

Runs on CPU (about 15-30 min per model here) or GPU if available. The multilingual model
is also scored on the original Portuguese MOZ messages (test name "moz_test_pt").
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from splits import ROOT, load_splits, save_preds


def batches(tok, texts, labels, max_len, bs, shuffle, seed=0):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([int(labels[k]) for k in j])
        yield enc


@torch.no_grad()
def predict(model, tok, texts, max_len, bs=64):
    model.eval()
    out = []
    for enc in batches(tok, texts, None, max_len, bs, shuffle=False):
        out.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--tag", default="distilbert")
    ap.add_argument("--train", choices=["real", "real+synth"], default="real+synth")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--save", action="store_true", help="save the fine-tuned model to models/<tag>")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="do not train the word-embedding matrix (saves memory on large-vocabulary models)")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    train, synth, tests = load_splits()
    tr = pd.concat([train, synth]) if args.train == "real+synth" else train
    texts, labels = tr.text.tolist(), tr.label.to_numpy()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if args.freeze_embeddings:
        emb = model.get_input_embeddings()
        emb.weight.requires_grad_(False)
        print(f"froze word embeddings ({emb.weight.numel():,} params)", flush=True)
    # Class-weighted loss, the transformer analogue of class_weight="balanced" in the baseline.
    w = torch.tensor(len(labels) / (2 * np.bincount(labels, minlength=2)), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(texts) / args.bs))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    t0, step = time.time(), 0
    for ep in range(args.epochs):
        model.train()
        for enc in batches(tok, texts, labels, args.max_len, args.bs, shuffle=True, seed=ep):
            y = enc.pop("labels")
            loss = loss_fn(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 25 == 0:
                print(f"epoch {ep + 1} step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)

    for name, t in tests.items():
        save_preds(args.tag, args.train, name, t, predict(model, tok, t.text.tolist(), args.max_len), 0.5)
        print(f"scored {name}", flush=True)
    if "multilingual" in args.model:
        moz = tests["moz_test"].assign(text=tests["moz_test"]["text_pt"])
        save_preds(args.tag, args.train, "moz_test_pt", moz, predict(model, tok, moz.text.tolist(), args.max_len), 0.5)
        print("scored moz_test_pt", flush=True)

    if args.save:
        out = ROOT / "models" / args.tag
        model.save_pretrained(out); tok.save_pretrained(out)
        print(f"saved {out}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
