# Detecting Mobile Money Scam SMS Without a Local Dataset

## 1. Problem

Mobile Money (MoMo) fraud in Ghana usually starts with an SMS. Common examples are a fake "you
received GHS 500" alert followed by a call asking for the money back, a threat that the wallet will
be blocked, a fake promotion that charges a "processing fee," or a request for a PIN or OTP. This
project builds a tool that flags these messages, explains why, and tells the user what to do next.

## 2. The data problem and how it was handled

No public dataset of Ghanaian MoMo scam SMS exists. Ghanaian news and telco warnings describe the
scams but rarely quote real messages. Real examples are mostly screenshots on social media.
Collecting them by hand was not possible for this project. So the research question became:

> Can a model trained on public English SMS data plus synthetic Ghana-style data catch **real**
> mobile-money scams it has never seen?

| Dataset | Content | Role |
|---|---|---|
| UCI SMS Spam Collection | 5,574 English SMS (spam/ham) | training + in-domain test |
| Mendeley SMS Phishing (Mishra & Soni, 2022, CC BY 4.0) | 5,971 English SMS (ham/spam/smishing) | training + in-domain test |
| Synthetic Ghana (`src/synth_ghana.py`) | 24 templates, 1,229 messages | **training only** |
| MOZ-Smishing (AfricaNLP 2025) | 2,561 real Mozambican SMS, 552 mobile-money scams, Portuguese | **test only**, machine-translated to English |
| Ghana challenge set (`src/ghana_challenge.py`) | 40 hand-written messages | **test only**, stress test |

A fifth candidate, `electricsheepafrica/africa-smishing-sms-phishing`, was rejected. Every row is
flagged synthetic, and the fields contradict each other (for example, English text for a Kenyan
bank priced in Nigerian naira, labelled "Swahili").

### Data handling
- **Scrubbing:** phone numbers, amounts, transaction IDs, names, links and dates become typed
  tokens (`<PHONE>`, `<AMOUNT>`, ...). Personal details are removed but the message structure is
  kept. Mobile-money brand names (M-Pesa, Telecel Cash, ...) are normalised to "MoMo", so a model
  does not treat Mozambican messages as different just because of the brand name.
- **Duplicates:** about 5,500 exact duplicates were removed, mostly because Mendeley re-uses UCI messages.
- **Campaign grouping:** messages with character TF-IDF cosine similarity ≥ 0.8 are grouped as
  one "campaign," and every split is made by group. This stops one scam template from appearing in
  both train and test.
- **No synthetic data in any test set.** A model scored on output from the same generator it
  trained on is only measuring memorisation.
- **MOZ dev/test:** MOZ is split by campaign group into a dev half (used for error analysis) and a
  test half (the reported number).

## 3. Methods

- **Rules:** 12 hand-written red flags (PIN/OTP request, "by mistake," "send it back," account
  threats, urgency, prize, fee-to-receive, link, call-this-number, verify-account, impersonation,
  USSD dial instruction).
- **TF-IDF + logistic regression:** word 1–2-grams plus character 3–5-grams, class-balanced.
- **DistilBERT** (`distilbert-base-uncased`): fine-tuned for 2 epochs, lr 5e-5, batch 32,
  max length 96, class-weighted loss, on the same splits, on CPU.

A third model, multilingual DistilBERT on the original Portuguese, was planned but could not
be completed: on the 16 GB machine used, with other jobs running, training crashed for lack of
memory three times (also with the embedding layer frozen). It can be run later with
`python src/train_transformer.py --model distilbert-base-multilingual-cased --tag mdistilbert --freeze-embeddings`
and `summarize.py` will add it to the table.

All models use a threshold of 0.5 (rules: score ≥ 2). 95% confidence intervals come from 1,000
bootstrap resamples of each test set.

## 4. Results

Main results (threshold 0.5; 95% bootstrap CIs in brackets; full table in `reports/results.md`).

| Test set | Model (training data) | Recall | FPR | Precision | PR-AUC |
|---|---|---|---|---|---|
| English held-out (1,295; 227 scams) | Rules only | 0.29 | 0.008 | 0.88 | 0.48 |
| | TF-IDF LR (real) | 0.95 | 0.008 | 0.96 | 0.99 |
| | TF-IDF LR (real + synth + spelling) | 0.96 | 0.005 | 0.98 | 0.99 |
| | DistilBERT (real + synth) | 0.98 | 0.006 | 0.97 | 0.99 |
| **MOZ test, real MoMo** (1,139; 273 scams) | Rules only | 0.01 | 0.014 | 0.14 | 0.23 |
| | TF-IDF LR (real) | 0.71 [0.66–0.77] | 0.39 [0.35–0.42] | 0.37 | 0.28 |
| | TF-IDF LR (real + synth) | 0.93 [0.90–0.96] | 0.29 [0.26–0.32] | 0.50 | 0.46 |
| | TF-IDF LR (real + synth + spelling) | 0.93 [0.90–0.96] | 0.28 [0.26–0.31] | 0.51 | 0.50 |
| | **DistilBERT (real + synth)** | **0.95 [0.92–0.97]** | **0.22 [0.19–0.25]** | **0.58** | **0.62** |
| Ghana challenge (40; 20 scams) | TF-IDF LR (real + synth + spelling) | 0.60 [0.36–0.81] | 0.15 | 0.80 | 0.84 |
| | DistilBERT (real + synth) | 0.75 [0.55–0.92] | 0.15 | 0.83 | 0.89 |

### Findings

1. **High scores on English data mean little.** Every trained model scores about 0.95 recall on the
   English held-out set. On real mobile-money SMS from another country, the same baseline drops to
   0.71 recall and flags 39% of genuine messages.
2. **Synthetic local data helps.** Adding the synthetic Ghana messages raised TF-IDF recall on real
   MOZ scams from 0.71 to 0.93 and cut false alarms from 39% to 29%. (See the disclosure in §6.)
3. **DistilBERT beats the baseline where it matters.** On English data every model is near the
   ceiling, so the comparison there says nothing. On real mobile-money SMS, DistilBERT catches
   slightly more scams (0.95 vs 0.93, overlapping CIs) and, more importantly, raises fewer false
   alarms (22% vs 28%; the CIs barely touch) with clearly better ranking (PR-AUC 0.62 vs 0.50). On
   the small Ghana challenge set its recall is 0.75 vs 0.60, but with 20 scams the CIs overlap
   widely. The website still uses TF-IDF LR: it runs instantly on any laptop, its red-flag
   explanations line up with its features, and the gap is modest.
4. **A text-only model cannot detect forged alerts.** A fake "Cash In received" alert reads exactly
   like a real one; only the *sender* differs (a personal number instead of the official ID). Most
   of the remaining false alarms on MOZ are genuine bank and M-Pesa transaction alerts, and some
   missed scams are forged "MoMo Confirmed ... balance blocked" alerts. Both the challenge-set
   forged alerts scored below 0.35. **A deployable tool must use the sender ID**, and none of the
   public datasets include it.
5. **Rules are a poor classifier but a good explainer.** Rules alone catch at most 29% of scams.
   But when a rule fires, its reason ("asks for your PIN") is literally true about the message. So
   the demo uses the model for the risk score and the rules for the explanation.
6. **Spelling mistakes are a weak signal in the available data.** A common belief is that scam
   SMS are badly spelled. `src/spelling.py` detects typos (an unknown word one edit from a real
   word: "recieved", "acount"), grammar slips ("You has received") and wrong-word errors ("Dear
   costumer"), while ignoring SMS shorthand and Ghanaian words ("pls", "abeg", "chale"). In the
   English data, scams and genuine messages trip it at almost the same rate (23.8% vs 24.7% on the
   held-out set; `reports/spelling_by_class.csv`), because everyday SMS is informal. On MOZ it cannot
   be tested fairly: machine translation fixes the original spelling. As model features it raised
   MOZ-test PR-AUC from 0.46 to 0.50 with unchanged recall. The model learned that *a* typo or
   grammar slip points to scam but *many* typos point to genuine (shorthand), see
   `reports/spelling_weights.json`. It is kept as a stated reason and a score boost, not as a
   verdict on its own. Only a real Ghanaian test set can settle how much it helps.
7. **No threshold fixes the false alarms.** Choosing the threshold on MOZ dev and testing on MOZ
   test (`reports/threshold_analysis.md`): holding false alarms to 10% leaves only 45% recall, and
   to 5% only 8%. The genuine alerts that get flagged score as high as real scams, so a threshold
   cannot separate them. The default 0.5 is kept (recall first, as planned).

### Sender check (product feature, not evaluated)
Because forged alerts can only be told apart by their sender, the website and API take an
optional **sender** as shown on the phone. If an alert-style message ("Payment received", "Current
Balance", "Transaction ID") comes from an ordinary phone number, it is flagged as almost certainly
fake (rule weight 4.0, risk 0.8). A message that claims to be from the provider (account threats,
PIN requests, prizes) from a phone number gets a smaller boost. A friend's message from a phone
number is unaffected. None of the public datasets record the sender, so this rule is tested only by
unit tests, not by the evaluation above.

### Error analysis (MOZ dev half and challenge set; `reports/error_analysis.json`)
- **Missed:** "my number has a problem, send the value to this other number in the name of X," which
  is the dominant Mozambican scam, plus forged balance-blocked alerts and translations too garbled
  to read.
- **False alarms:** genuine bank and wallet transaction alerts, operator service notices, and
  operator *anti-fraud warnings* (which use the same vocabulary as scams).
- **Challenge set:** fails on "lost my phone, this is my new number" impersonation and on scams with
  no trigger words. It falsely flags MTN's own "we will never ask for your PIN" warning and a
  genuine "wallet locked after wrong PIN attempts" notice.

## 5. Website

`python run_site.py` starts a local website at http://localhost:8765 (FastAPI serving
`web/index.html`; the Python model runs on the same machine, nothing goes online). For each message
it shows a risk level and score, the red flags found, concrete next steps (check your balance on the
official menu, *170# on MTN; never share your PIN or OTP; report to 1515), the wording that raised the
score (with a note that highlights are not proof), and the scrubbed text the model saw. It also has
the optional sender field, a four-step "before you send money back" checklist, and a plain account of
accuracy and limits. `POST /api/predict {"text", "sender"}` exposes the same logic. A Streamlit
version (`streamlit run app.py`) is kept as an alternative.

## 6. Limitations and honest caveats

- **No real Ghanaian scam SMS in any test set.** MOZ is real mobile-money fraud, but it is from
  Mozambique and the English version is machine-translated, often badly (M-Pesa became
  "M-weight" before normalisation). The Ghana challenge set is constructed and only 40 messages,
  so its confidence intervals are wide.
- **Not fully blind:** the "send it to my other number" synthetic template was added after looking
  at errors across the whole MOZ set (v1). Only after that was MOZ split into dev and test. v1
  without the template scored recall 0.73 and FPR 0.31 on all of MOZ.
- The synthetic genuine-alert templates approximate MTN and Telecel formats. They are not official templates.
- Labels: spam and smishing are merged into "scam." Some UCI "spam" is marketing rather than fraud.
- SMS only. Many MoMo scams are carried out by phone call.

## 7. What would make this deployable

1. Read the **sender ID** automatically (for example, as an Android SMS app) instead of asking the
   user to type it, and measure the sender rule on real labelled messages.
2. Collect a few hundred **real Ghanaian** scam and genuine messages, with consent, to use as a
   blind test set, for example through a telco or the Cyber Security Authority.
3. Re-check precision on genuine operator notices and fraud warnings, the main false-alarm source.

## Reproduce

See `README.md`. Results tables: `reports/results.md` and `reports/results.csv`. Per-message
predictions: `reports/preds/`.
