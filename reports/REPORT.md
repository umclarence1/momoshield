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
- **DistilBERT** (`distilbert-base-uncased`) and **multilingual DistilBERT**
  (`distilbert-base-multilingual-cased`): fine-tuned for 2 epochs, lr 5e-5, class-weighted loss,
  on the same splits. The multilingual model is also tested on the **original Portuguese** MOZ
  messages, with no translation step.

All models use a threshold of 0.5 (rules: score ≥ 2). 95% confidence intervals come from 1,000
bootstrap resamples of each test set.

## 4. Results

RESULTS_TABLE

### Findings

1. **High scores on English data mean little.** Every trained model scores about 0.95 recall on the
   English held-out set. On real mobile-money SMS from another country, the same baseline drops to
   0.71 recall and flags 39% of genuine messages.
2. **Synthetic local data helps.** Adding the synthetic Ghana messages raised TF-IDF recall on real
   MOZ scams from 0.71 to 0.93 and cut false alarms from 39% to 29%. (See the disclosure in §6.)
3. **Bigger models.** TRANSFORMER_FINDING
4. **A text-only model cannot detect forged alerts.** A fake "Cash In received" alert reads exactly
   like a real one; only the *sender* differs (a personal number instead of the official ID). Most
   of the remaining false alarms on MOZ are genuine bank and M-Pesa transaction alerts, and some
   missed scams are forged "MoMo Confirmed ... balance blocked" alerts. Both the challenge-set
   forged alerts scored below 0.35. **A deployable tool must use the sender ID**, and none of the
   public datasets include it.
5. **Rules are a poor classifier but a good explainer.** Rules alone catch at most 20% of scams.
   But when a rule fires, its reason ("asks for your PIN") is literally true about the message. So
   the demo uses the model for the risk score and the rules for the explanation.

### Error analysis (MOZ dev half and challenge set; `reports/error_analysis.json`)
- **Missed:** "my number has a problem, send the value to this other number in the name of X," which
  is the dominant Mozambican scam, plus forged balance-blocked alerts and translations too garbled
  to read.
- **False alarms:** genuine bank and wallet transaction alerts, operator service notices, and
  operator *anti-fraud warnings* (which use the same vocabulary as scams).
- **Challenge set:** fails on "lost my phone, this is my new number" impersonation and on scams with
  no trigger words. It falsely flags MTN's own "we will never ask for your PIN" warning and a
  genuine "wallet locked after wrong PIN attempts" notice.

## 5. Demo

`streamlit run app.py` shows a risk level, the rule-based **reasons for concern**, concrete **next
steps** (check your balance on the official menu, *170# on MTN; never share your PIN or OTP; report to
1515), and the words that pushed the model towards "scam," with a note that highlights are not proof.
`uvicorn api:app --app-dir src` serves the same logic as `POST /predict`.

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

1. Use **sender ID** as a feature: official short code vs. personal number. This alone addresses the forged-alert problem.
2. Collect a few hundred **real Ghanaian** scam and genuine messages, with consent, to use as a
   blind test set, for example through a telco or the Cyber Security Authority.
3. Re-check precision on genuine operator notices and fraud warnings, the main false-alarm source.

## Reproduce

See `README.md`. Results tables: `reports/results.md` and `reports/results.csv`. Per-message
predictions: `reports/preds/`.
