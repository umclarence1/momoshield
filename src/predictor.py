"""Shared prediction logic for the API and the Streamlit demo."""
from pathlib import Path

import joblib

import rules
from scrub import scrub
from text_model import TfidfLR  # noqa: F401  (needed to unpickle the model)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tfidf_lr_spelling.joblib"

ADVICE = {
    "mistaken_transfer": "Check your real balance and history through the official MoMo menu (*170# on MTN) or the official app. A genuine wrong transfer is reversed by the network, never by you sending money.",
    "send_back": "Do not send money just because a message or caller asks you to. Check your real transaction history first.",
    "pin_otp_request": "Never share your PIN or OTP with anyone, including people who say they work for MTN or Telecel.",
    "dial_code": "Do not dial a code or approve a prompt that a stranger tells you to. It may authorise a payment out of your wallet.",
    "account_threat": "Providers do not close accounts by SMS from a personal number. Call your provider's official customer care line yourself, using a number from their official website.",
    "verify_account": "Update your details only at an official service centre or in the official app, never through a link or a number in an SMS.",
    "prize_reward": "Genuine promotions never ask you to pay a fee or share a PIN to collect a prize.",
    "fee_to_receive": "Paying a fee to receive money is a classic scam pattern.",
    "link": "Do not open links in unexpected SMS messages.",
    "spelling": "Poor spelling is a warning sign in a message about money. Genuine provider messages are carefully written.",
    "alert_from_phone": "Ignore this alert. Check your balance on the official MoMo menu: if the money is not there, it never arrived. Do not send anything back.",
    "provider_from_phone": "Do not reply or call back. Contact your provider through its official channels if you are worried about your account.",
}
# Placeholder tokens as the word vectoriser sees them (it strips the angle brackets).
PLACEHOLDER_WORDS = {"phone", "amount", "num", "txid", "url", "email", "name", "date", "time"}
DEFAULT_ADVICE = "If in doubt, check your balance through the official MoMo menu and report suspicious messages to your provider (MTN: 1515)."


class Predictor:
    def __init__(self, path=MODEL_PATH):
        self.model = joblib.load(path)

    def analyse(self, text: str, sender: str | None = None) -> dict:
        clean = scrub(text)
        p = float(self.model.predict_proba([clean])[0])
        rule_score, hits = rules.check(clean, sender)
        # Combine: the model score, nudged up when hard red flags fire.
        risk = max(p, min(1.0, rule_score / 5))
        level = "high" if risk >= 0.7 else "medium" if risk >= 0.4 else "low"
        # Sender-based advice is the most specific, so it goes first.
        hits_for_advice = sorted(hits, key=lambda h: not h.name.endswith("_from_phone"))
        advice = [ADVICE[h.name] for h in hits_for_advice if h.name in ADVICE] or [DEFAULT_ADVICE]
        return {
            "risk": round(risk, 3),
            "level": level,
            "model_probability": round(p, 3),
            "reasons": [h.reason for h in hits],
            "advice": list(dict.fromkeys(advice))[:3],
            "influential_words": [w for w, c in self.model.top_words(clean, k=15)
                                  if c > 0 and len(w) > 2 and not set(w.split()) & PLACEHOLDER_WORDS][:6],
            "scrubbed_text": clean,
            "sender_kind": rules.sender_kind(sender),
        }
