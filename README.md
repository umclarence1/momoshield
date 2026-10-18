# MoMo Scam SMS Detector

Flags Mobile Money scam SMS (fake "sent by mistake" alerts, blocked-account threats, fake promos,
PIN/OTP theft), explains why, and tells the user what to do next.

## Research question

No public dataset of Ghanaian MoMo scam SMS exists. So the question is:

> Can a model trained on public English SMS data plus synthetic Ghana-style data
> catch **real** mobile-money scams it has never seen?

## Data

| Set | Source | Use |
|---|---|---|
| UCI SMS Spam | `ucirvine/sms_spam` (Hugging Face) | train / in-domain test |
| Mendeley SMS Phishing (CC BY 4.0) | doi:10.17632/f45bkkt8pr.1 | train / in-domain test |
| Synthetic Ghana | `src/synth_ghana.py` | **training only** |
| MOZ-Smishing (real Mozambican MoMo SMS) | `MOZNLP/MOZ-Smishing`, machine-translated pt→en | **test only** |
| Ghana challenge set | `src/ghana_challenge.py`, hand-written, 40 msgs | **test only**, stress test |

Not used: `electricsheepafrica/africa-smishing-sms-phishing`. Every row is synthetic and the fields
are inconsistent (e.g. English Equity Bank text in NGN, labelled "Swahili", country "Nigeria").

Data rules:
- Phone numbers, amounts, IDs, names, links and dates become typed tokens (`<PHONE>`, `<AMOUNT>`,
  `<TXID>`, `<NAME>`, `<URL>`) so the structure stays and personal details go (`src/scrub.py`).
- Near-duplicate messages (char TF-IDF cosine ≥ 0.8) are grouped into "campaigns" and the
  train/test split is made by group, so one scam template never sits on both sides.
- Synthetic and constructed data never go into a test set that is reported as real.
- Label: spam + smishing = 1, ham/legitimate = 0.

## Run

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python src/download_data.py          # HF datasets -> data/raw
# Mendeley: download Dataset_5971.zip into data/raw/mendeley and unzip
python src/translate_moz.py          # MOZ-Smishing pt -> en (CPU, ~10 min)
python src/synth_ghana.py
python src/ghana_challenge.py
python src/prepare_data.py
python src/train_eval.py             # results -> reports/
streamlit run app.py                 # demo
uvicorn api:app --app-dir src        # API: POST /predict {"text": "..."}
```

## Baseline results (v2)

TF-IDF + logistic regression, threshold 0.5. Full table: `reports/baseline_results.csv`.

| Test set | Train data | Recall | Precision | False-positive rate |
|---|---|---|---|---|
| English held-out | real | 0.95 | 0.96 | 0.008 |
| English held-out | real + synthetic Ghana | 0.95 | 0.99 | 0.003 |
| **MOZ real MoMo (test half)** | real | 0.71 | 0.37 | 0.39 |
| **MOZ real MoMo (test half)** | real + synthetic Ghana | **0.93** | 0.50 | 0.29 |
| Ghana challenge (40, constructed) | real + synthetic Ghana | 0.60 | 0.80 | 0.15 |
| any | rules only | ≤ 0.20 | — | — |

What this shows:
- Near-perfect scores on English data say little. On real mobile-money SMS from another country,
  recall drops to 0.71 and almost 4 in 10 genuine messages get flagged.
- Adding synthetic Ghana data raises recall on real MoMo scams from 0.71 to 0.93 and lowers false alarms.
- The remaining false alarms are mostly **genuine transaction alerts**. A forged cash-in alert
  has the same text as a real one; only the sender differs. Text alone cannot separate them,
  so a deployable tool needs the sender ID.
- The rules are a weak classifier (recall ≤ 0.20) but an accurate source of "reasons for concern."

**Disclosure:** the "send it to my other number" synthetic template was added after looking at
errors on the full MOZ set (v1). After that, MOZ was split by campaign group into dev (for error
analysis) and test (reported). So the MOZ-test figure is not fully blind. v1 (before the template)
scored recall 0.73 and FPR 0.31 on all of MOZ.

## Limitations

- No real Ghanaian scam SMS in the test data. MOZ-Smishing is real mobile-money fraud, but from
  another country and machine-translated.
- The synthetic genuine-alert templates are approximations of MTN/Telecel formats, not official templates.
- The model only sees the text. Sender ID, which is one of the strongest real-world signals, is not available in the public data.
- Many MoMo scams happen by phone call. This tool only covers SMS.
