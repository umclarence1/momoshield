"""Replace personal / variable details in SMS text with typed placeholder tokens.

Removing numbers outright would throw away structure that matters (a scam
"from" a personal phone number vs. a genuine alert with a transaction ID),
so each kind of detail is replaced by its own token instead.
"""
import re

CURRENCY = r"(?:GH[S₵C¢]|GH\s?₵|GHC|GHS|₵|¢|cedis?|MZN|MTS?|MT|KES|KSH|NGN|₦|ZAR|AUD|USD|Rs\.?|INR|£|\$|€)"
AMOUNT_NUM = r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

_PATTERNS = [
    ("<URL>", re.compile(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|ly|gh|co|info|xyz|link|top|site|online)(?:/\S*)?\b", re.I)),
    ("<EMAIL>", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("<AMOUNT>", re.compile(rf"{CURRENCY}\s?(?:{AMOUNT_NUM})|(?:{AMOUNT_NUM})\s?{CURRENCY}(?![a-z])", re.I)),
    ("<DATE>", re.compile(r"\b\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?\b")),
    ("<TIME>", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AP]M)?\b", re.I)),
    # +233 24 412 3456, 0244123456, 024-412-3456, 258849018210, 07808726822
    ("<PHONE>", re.compile(r"(?<![\w])(?:\+?\d{1,3}[\s-]?)?\(?0?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{3,4}(?![\w])")),
    # Transaction / reference IDs: mixed letters+digits of length >= 6, or long digit runs.
    ("<TXID>", re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6,}\b|\b\d{8,}\b")),
    ("<NUM>", re.compile(r"\b\d+(?:[.,]\d+)*\b")),
]

# Upper-case personal names in alert-style messages: "from KOFI MENSAH", "- RAIMUNDO JOSE"
_NOT_NAMES = r"(?!(?:MTN|TELECEL|VODAFONE|AIRTELTIGO|AT|GCB|ECG|DSTV|BOLT|MOMO|GHS|GRA|SSNIT|NHIS)\b)"
_NAME = re.compile(rf"(?<=\bfrom\s){_NOT_NAMES}(?:[A-Z][A-Z'.-]+\s?){{1,4}}(?=\W|$)"
                   rf"|(?<=\bto\s){_NOT_NAMES}(?:[A-Z][A-Z'.-]+\s){{1,3}}[A-Z][A-Z'.-]+(?=\W|$)"
                   r"|(?<=<PHONE> - )(?:[A-Z][A-Z'.-]+\s?){1,4}")
# USSD codes like *170# or *170*1# are not personal data and are a useful signal.
_USSD = re.compile(r"\*\d+(?:\*\d+)*#")
_SPACES = re.compile(r"\s+")
# IDs introduced by a label are IDs even when they look like phone numbers.
_LABELLED_ID = re.compile(r"\b((?:financial\s+)?(?:transaction|trans|txn)\s*id|ref(?:erence)?(?:\s*no)?)\s*[:.]?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b", re.I)


# Mobile-money brands from different countries mean the same thing to a scam detector.
# "M-weight" is how the pt->en translator mangles "M-Pesa" ("pesa" = "weighs").
_MM_BRAND = re.compile(r"\b(m[-_ ]?pesa|m[-_ ]?weight|mpesa|e-?mola|mkesh|ponto ?24|point ?24|p24|mobile money|momo|telecel cash|vodafone cash|airteltigo money|at money)\b", re.I)


def scrub(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _MM_BRAND.sub("MoMo", text)
    text = _LABELLED_ID.sub(r"\1: <TXID>", text)
    ussd = _USSD.findall(text)
    text = _USSD.sub("\x00", text)
    for token, pat in _PATTERNS:
        text = pat.sub(token, text)
    for code in ussd:
        text = text.replace("\x00", code, 1)
    text = _NAME.sub("<NAME> ", text)
    return _SPACES.sub(" ", text).strip()


if __name__ == "__main__":
    samples = [
        "Payment received for GHS 50.00 from KOFI MENSAH Current Balance: GHS 120.50. Transaction ID: 41234567890.",
        "You have received GH₵500 by mistake, send back to 0244123456 now or your account will be blocked",
        "Confirmado 7FB9483TA4V. Recebeste 1,000.00MT de 258849018210 - RAIMUNDO JOSE aos 11/6/20 as 8:48 AM.",
        "URGENT! call 0871-872-9758 or visit www.claim-prize.com",
    ]
    for s in samples:
        print(scrub(s))
