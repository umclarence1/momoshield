"""Spelling and grammar red flags.

Ghanaian fraud warnings point out that scam SMS are often badly written ("You has received",
"recieved", "acount"), while official alerts are machine-generated and spelled correctly.
But friends' texts are informal too ("pls", "u", "abeg"), so this module separates:
  - typos: an unknown word that is one edit away from a real English word ("recieved")
  - grammar slips: specific broken patterns ("you has", "have been credit")
from SMS shorthand and Ghanaian / Pidgin words, which are not counted.
"""
import re
from functools import lru_cache

from spellchecker import SpellChecker

_SPELL = SpellChecker(distance=1)

# Words that are fine in a Ghanaian SMS even though an English dictionary does not know them.
ALLOW = set("""
momo mtn telecel vodafone airteltigo ecg dstv gcb ghs cedi cedis pesewas ussd otp sim kyc
whatsapp bolt jumia hubtel glovo ecobank fidelity stanbic calbank absa mpesa ghana ghanaian
accra kumasi tema takoradi tamale kaneshie madina circle osu legon korle bu achimota spintex
abeg chale charley chaley oo ooo paa dey wey eiii eii ei herh ahh akwaaba medaase meda ase
ampa kwasia waakye kenkey banku fufu kelewele trotro obroni obruni bra bro sis wofa auntie
kofi ama kwame akosua yaw abena kwabena adwoa kojo efua kwesi kweku akua esi nana yaa afua
mensah owusu boateng asante osei appiah addo amoah darko quaye tetteh ansah amponsah sarpong
agyemang forson nyarko boakye adjei arthur issah abubakar
pls plz plss u ur urs r n b4 2day 2moro tmrw tmr msg msgs txt thx thnx tnx k kk ok okay
lol pic pics bday gonna wanna gotta luv hav dat dis den wat wats coz cos bcos bcoz abt
im ive ill id dont cant wont didnt isnt wasnt doesnt aint youre theyre thats whats lets
""".split())

GRAMMAR = [
    re.compile(p, re.I) for p in [
        r"\b(you|we|they|i) (has|was been|is been)\b",
        r"\b(he|she|it|account|wallet|number) (have)\b(?! to)",
        r"\b(has|have|had) been (credit|debit|transfer|block|suspend|send|receive)\b",
        r"\b(have|has) (send|receive|block|transfer)\b",
        r"\byou (receive|send) (money|cash|amount|<AMOUNT>) (from|to)\b(?=.*\b(mistake|wrong))",
        r"\b(a|an) (amount|money) (of )?<AMOUNT> (have)\b",
        r"\bwrongly mistake\b|\bmistakenly by mistake\b",
        r"\b(kindly|please) to\b",
        # Real words used wrongly: a dictionary cannot catch these.
        r"\b(dear|valued|esteemed) costumers?\b",
        r"\bcongratulation\b(?!s)",
        r"\bloose (your|the|money|access|account)\b",
        r"\b(you|your) (wins|winned)\b",
    ]
]
_WORD = re.compile(r"(?<![<\w])[A-Za-z][a-z']{2,}(?![\w>])")
_SENT_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-z']+)")


@lru_cache(maxsize=50_000)
def _is_typo(word: str) -> bool:
    w = word.lower().strip("'")
    if len(w) < 4 or w in ALLOW or w in _SPELL:
        return False
    # Unknown and one edit from a real word -> a typo. Far from any word -> probably a
    # name or a local word, which we do not count.
    cand = _SPELL.correction(w)
    return bool(cand) and cand != w and cand in _SPELL


def find_typos(text: str):
    """Misspelled words, skipping placeholders and capitalised names mid-sentence."""
    starts = {m.start(1) for m in _SENT_START.finditer(text)}
    out = []
    for m in _WORD.finditer(text):
        w = m.group(0)
        if w[0].isupper() and m.start() not in starts:
            continue  # probably a proper noun
        if _is_typo(w):
            out.append(w)
    return out


def find_grammar(text: str):
    return [m.group(0) for p in GRAMMAR for m in [p.search(text)] if m]


def features(text: str):
    """Dense features for the model: typo rate, typo present, grammar slips, odd capitalisation."""
    words = _WORD.findall(text)
    n = max(1, len(words))
    typos = find_typos(text)
    grammar = find_grammar(text)
    caps_words = sum(1 for w in words if len(w) > 2 and w[0].isupper())
    return [
        min(1.0, len(typos) / n * 5),            # typo density, saturating at 1 typo per 5 words
        1.0 if typos else 0.0,
        min(1.0, len(grammar) / 2),
        1.0 if n >= 6 and caps_words / n > 0.6 else 0.0,  # Title Case Every Word
        1.0 if re.search(r"[a-z][,.][A-Za-z]{2,}", text) else 0.0,  # missing space after punctuation
    ]


FEATURE_NAMES = ["typo_density", "has_typo", "grammar_slips", "title_case", "missing_space"]
