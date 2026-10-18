import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import rules  # noqa: E402
from scrub import scrub  # noqa: E402


def test_scrub_replaces_personal_details():
    s = scrub("Payment received for GHS 50.00 from KOFI MENSAH Current Balance: GHS 120.50. Transaction ID: 41234567890.")
    assert "KOFI" not in s and "MENSAH" not in s and "41234567890" not in s
    assert "<AMOUNT>" in s and "<NAME>" in s and "<TXID>" in s


def test_scrub_phone_url_date():
    s = scrub("Call 0244123456 or visit www.claim-prize.com before 05/11")
    assert s == "Call <PHONE> or visit <URL> before <DATE>"


def test_scrub_keeps_ussd_and_brands():
    assert scrub("Dial *170# now") == "Dial *170# now"
    assert "MTN" in scrub("I am calling from MTN, a code will come")


def test_scrub_normalises_mobile_money_brands():
    assert scrub("send via M-Pesa") == "send via MoMo"
    assert scrub("Telecel Cash: received") == "MoMo: received"


def test_rules_flag_mistaken_transfer_scam():
    score, hits = rules.check(scrub("You have received GHS 500 by mistake. Send it back to 0244123456 immediately or your account will be blocked."))
    names = {h.name for h in hits}
    assert score >= rules.THRESHOLD
    assert {"mistaken_transfer", "send_back", "account_threat", "urgency"} <= names


def test_rules_quiet_on_genuine_alert():
    score, _ = rules.check(scrub("Payment received for GHS 120.00 from AKUA DARKO Current Balance: GHS 340.55. Transaction ID: 51234098765."))
    assert score < rules.THRESHOLD


def test_spelling_flags_scam_typos_and_grammar():
    import spelling
    t = scrub("You has recieved GHS 500 by mistake, send it back to 0244123456")
    assert "recieved" in spelling.find_typos(t)
    assert spelling.find_grammar(t)
    assert "spelling" in {h.name for h in rules.check(t)[1]}
    assert spelling.find_grammar("Dear costumer your account is blocked")


def test_spelling_ignores_sms_shorthand_slang_and_alerts():
    import spelling
    for t in ["Abeg buy me pure water on your way chale",
              "pls send me 50 for transport, ill pay u back friday",
              "Payment received for GHS 120.00 from AKUA DARKO Current Balance: GHS 340.55. Transaction ID: 51234098765."]:
        c = scrub(t)
        assert not spelling.find_typos(c) and not spelling.find_grammar(c), t
