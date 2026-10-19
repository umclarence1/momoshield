import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if not (Path(__file__).resolve().parent.parent / "models" / "tfidf_lr_spelling.joblib").exists():
    pytest.skip("trained model not present (run src/train_eval.py)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)
ALERT = ("Payment received for GHS 120.00 from AKUA DARKO Current Balance: GHS 340.55. "
         "Reference: lunch. Transaction ID: 51234098765. TRANSACTION FEE: 0.00")


def predict(text, sender=None):
    r = client.post("/api/predict", json={"text": text, "sender": sender})
    assert r.status_code == 200
    return r.json()


def test_homepage_served():
    r = client.get("/")
    assert r.status_code == 200 and "MoMo Scam Check" in r.text


def test_obvious_scam_is_high_with_reasons():
    r = predict("You have received GHS 500 by mistake. Send it back to 0244123456 immediately or your account will be blocked.")
    assert r["level"] == "high" and len(r["reasons"]) >= 3 and r["advice"]


def test_genuine_alert_from_official_sender_is_low():
    r = predict(ALERT, "MobileMoney")
    assert r["level"] == "low" and r["sender_kind"] == "name"


def test_same_alert_from_phone_number_is_flagged():
    r = predict(ALERT, "0551 234 567")
    assert r["level"] == "high" and r["sender_kind"] == "phone"
    assert any("ordinary phone number" in x for x in r["reasons"])


def test_friend_from_phone_number_is_not_flagged_for_sender():
    r = predict("Did you receive the GHS 80? Network was bad so I'm not sure it went", "0244123456")
    assert r["level"] == "low"


def test_input_validation():
    assert client.post("/api/predict", json={"text": ""}).status_code == 422
    assert client.post("/api/predict", json={"text": "a" * 2001}).status_code == 422
