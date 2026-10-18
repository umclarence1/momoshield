"""Hand-written red-flag rules.

Used two ways: as a rules-only baseline classifier, and to produce the
human-readable "reasons for concern" shown in the demo (these are always
literally true about the message, unlike word-importance highlights).
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    reason: str
    pattern: re.Pattern
    weight: float = 1.0


def _r(p):
    return re.compile(p, re.I)


RULES = [
    Rule("pin_otp_request", "Asks for your PIN, OTP or verification code. No genuine provider will ask for these.",
         _r(r"\b(send|share|give|enter|reply|tell|provide|confirm|dial)\b.{0,40}\b(pin|otp|one[- ]time|verification code|secret code|code)\b|\b(pin|otp)\b.{0,30}\b(send|share|reply|confirm)"), 2.0),
    Rule("mistaken_transfer", "Claims money was sent to you by mistake. This is the most common MoMo scam script.",
         _r(r"\b(by mistake|mistakenly|wrong(ly)? (number|transfer|transaction|account)|in error|erroneous(ly)?)\b"), 2.0),
    Rule("send_back", "Asks you to send or return money.",
         _r(r"\b(send|return|refund|reverse|transfer)\b.{0,25}\b(it |the money |the amount )?back\b|\b(kindly|please|pls)\b.{0,20}\b(send|return|refund)\b"), 1.5),
    Rule("account_threat", "Threatens to block, suspend or close your account.",
         _r(r"\b(block(ed)?|suspend(ed)?|deactivat\w*|clos(e|ed)|frozen|freeze|restrict\w*)\b.{0,40}\b(account|wallet|sim|line|number|momo)\b|\b(account|wallet|sim|line|momo)\b.{0,40}\b(block(ed)?|suspend(ed)?|deactivat\w*|frozen|restrict\w*)\b"), 1.5),
    Rule("urgency", "Pressures you to act immediately.",
         _r(r"\b(urgent(ly)?|immediately|right now|asap|within \S+ (hours?|minutes?|mins?)|today only|act now|expires?|last chance|now or)\b"), 1.0),
    Rule("prize_reward", "Claims you have won a prize, promotion or reward.",
         _r(r"\b(you('ve| have)? (won|been selected)|winner|congratulations|congrats|prize|lottery|jackpot|promo(tion)? (winner|reward)|claim your|bonus (of|reward))\b"), 1.5),
    Rule("fee_to_receive", "Asks you to pay a fee before you can receive money or a reward.",
         _r(r"\b(processing|registration|activation|clearance|release|delivery|customs|service) (fee|charge)\b|\bpay\b.{0,30}\b(to (receive|claim|release|unlock))\b"), 1.5),
    Rule("link", "Contains a link. Genuine MoMo alerts do not ask you to click links.",
         _r(r"<URL>|https?://|www\."), 1.0),
    Rule("call_number", "Asks you to call or contact a phone number.",
         _r(r"\b(call|contact|whatsapp|text|sms|reach)\b.{0,20}(<PHONE>|\b0\d{9}\b)"), 1.0),
    Rule("verify_account", "Asks you to verify, update or reactivate your account details.",
         _r(r"\b(verify|update|confirm|validate|reactivate|unblock|kyc)\b.{0,30}\b(account|details|information|identity|ghana card|wallet|sim)\b"), 1.5),
    Rule("impersonation", "Claims to come from a provider, agent or customer care.",
         _r(r"\b(customer (care|service)|mtn|telecel|vodafone|airteltigo|momo (agent|office|team)|mobile money (office|team)|head office|security team|fraud (team|department))\b"), 0.5),
    Rule("dial_code", "Tells you to dial a USSD code (scammers use this to make you authorise payments).",
         _r(r"\b(dial|press|enter)\b.{0,15}\*\d+"), 1.5),
]

THRESHOLD = 2.0


def check(text: str):
    """Return (score, list of triggered Rule)."""
    hits = [r for r in RULES if r.pattern.search(text or "")]
    return sum(r.weight for r in hits), hits


def predict_score(texts):
    return [check(t)[0] for t in texts]


if __name__ == "__main__":
    for msg in [
        "You have received <AMOUNT> by mistake. Send it back to <PHONE> immediately or your account will be blocked.",
        "Payment received for <AMOUNT> from <NAME> Current Balance: <AMOUNT>. Transaction ID: <TXID>.",
        "Hey, are we still meeting at 4? I'll send you the money for the tickets later",
    ]:
        s, hits = check(msg)
        print(f"{s:.1f}", [h.name for h in hits], "|", msg)
