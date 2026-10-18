"""Generate synthetic Ghana-style MoMo SMS for TRAINING ONLY.

Genuine alert templates approximate the shape of MTN MoMo / Telecel Cash
notifications (amount, counterparty, balance, transaction ID, fee). They are
reconstructions, not copies of official templates: swap in real ones if you
get them. Scam templates follow patterns described by Ghanaian news, the
Cyber Security Authority and telco warnings (mistaken-transfer, blocked
account, fake promo, PIN/OTP theft, fake agent/customer care).

Never put this data in a test set: a model scored on text from the same
generator it trained on is only measuring memorisation of these templates.
Every row carries campaign_id = template id so splits can be grouped.
"""
import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "synthetic_ghana.csv"

FIRST = ["KOFI", "AMA", "KWAME", "AKOSUA", "YAW", "ABENA", "KWABENA", "ADWOA", "KOJO", "EFUA", "ISAAC", "GRACE",
         "EMMANUEL", "MERCY", "SAMUEL", "PATIENCE", "IBRAHIM", "FATIMA", "MICHAEL", "VIDA", "DANIEL", "ESTHER"]
LAST = ["MENSAH", "OWUSU", "BOATENG", "ASANTE", "OSEI", "APPIAH", "ADDO", "AMOAH", "DARKO", "QUAYE", "TETTEH",
        "ANSAH", "AMPONSAH", "SARPONG", "ABUBAKAR", "ISSAH", "AGYEMANG", "FORSON"]
MERCHANTS = ["SHOPRITE ACCRA MALL", "MELCOM", "ECG POSTPAID", "GHANA WATER", "DSTV GHANA", "KFC OSU", "BOLT GH",
             "MAX MART", "TOTAL ENERGIES", "PALACE MALL", "UBER GHANA", "JUMIA GH"]
PREFIXES = ["024", "054", "055", "059", "020", "050", "027", "057", "026", "056"]


def name():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def phone():
    return random.choice(PREFIXES) + "".join(random.choice("0123456789") for _ in range(7))


def amt(lo=1, hi=3000):
    v = random.choice([random.randint(lo, hi), round(random.uniform(lo, hi), 2), random.choice([20, 50, 100, 200, 500, 1000])])
    return f"{v:,.2f}"


def cur():
    return random.choice(["GHS", "GHS ", "GH¢", "GHC "])


def txid():
    return "".join(random.choice("0123456789") for _ in range(random.choice([10, 11, 12])))


def date():
    return f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.choice([2024, 2025, 2026])}"


def time_():
    return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"


def maybe(s, p=0.5):
    return s if random.random() < p else ""


# ---------------------------------------------------------------- genuine
GENUINE = {
    "g_cashin_mtn": lambda: f"Payment received for {cur()}{amt()} from {name()} Current Balance: {cur()}{amt()} . Available Balance: {cur()}{amt()}. Reference: {random.choice(['1', 'rent', 'food', 'school fees', 'thanks', ''])}. Transaction ID: {txid()}. TRANSACTION FEE: 0.00",
    "g_transfer_out_mtn": lambda: f"Payment made for {cur()}{amt()} to {name()} {phone()}. Current Balance: {cur()}{amt()}. Available Balance: {cur()}{amt()}. Reference: {random.choice(['1', 'transport', 'chop money', 'loan'])}. Transaction ID: {txid()}. Fee charged: {cur()}{amt(0, 15)} Tax charged: {cur()}0.00",
    "g_cashout": lambda: f"Cash Out made for {cur()}{amt()} to {random.choice(MERCHANTS[:3] + ['AGENT ' + name()])}, Current Balance: {cur()}{amt()} Financial Transaction Id: {txid()}. Fee charged: {cur()}{amt(0, 20)}",
    "g_merchant": lambda: f"Payment for {cur()}{amt()} to {random.choice(MERCHANTS)} was successful. Current Balance: {cur()}{amt()}. Transaction ID: {txid()}. Fee: {cur()}0.00",
    "g_airtime": lambda: f"You have bought {cur()}{amt(1, 100)} airtime for {random.choice(['yourself', phone()])}. Your new balance is {cur()}{amt()}. Transaction ID: {txid()}",
    "g_bundle": lambda: f"You have successfully purchased {random.choice(['1.5GB', '3GB', '6.5GB', '10GB', '500MB'])} data bundle valid for {random.choice(['24 hours', '7 days', '30 days'])}. Dial *138# to check your balance.",
    "g_telecel_cash": lambda: f"Telecel Cash: You have received {cur()}{amt()} from {name()} on {date()} at {time_()}. Your new balance is {cur()}{amt()}. Transaction ID: {txid()}.",
    "g_telecel_sent": lambda: f"Telecel Cash: {cur()}{amt()} sent to {name()} {phone()} on {date()}. Fee {cur()}{amt(0, 10)}. New balance {cur()}{amt()}. Ref {txid()}.",
    # Genuine messages that contain "scary" words: reversal, failure, PIN reminders, real OTPs.
    "g_reversal": lambda: f"Your transaction of {cur()}{amt()} with ID {txid()} has been reversed. Your new balance is {cur()}{amt()}. Thank you for using MoMo.",
    "g_failed": lambda: f"Transaction failed. Your payment of {cur()}{amt()} to {random.choice(MERCHANTS)} could not be completed. You have not been debited. Transaction ID: {txid()}.",
    "g_pin_warning": lambda: random.choice([
        "MTN will NEVER call you to ask for your MoMo PIN. Do not share your PIN with anyone. Report fraud to 1515.",
        "Beware of fraudsters! Telecel staff will never ask for your PIN or OTP. Stay safe.",
        "Dear customer, never share your Mobile Money PIN or OTP with anyone, including people claiming to be MTN staff.",
    ]),
    "g_otp": lambda: f"{random.choice(['Your', 'Use'])} {random.choice(['Bolt', 'Jumia', 'Ecobank', 'GCB', 'Hubtel', 'Glovo'])} verification code is {random.randint(1000, 999999)}. {random.choice(['Do not share this code with anyone.', 'It expires in 10 minutes.', ''])}",
    "g_bank": lambda: f"{random.choice(['GCB', 'Ecobank', 'Fidelity', 'Stanbic', 'CalBank'])}: Acct **{random.randint(1000, 9999)} {random.choice(['debited', 'credited'])} with GHS{amt()} on {date()}. Desc: {random.choice(['MOMO TRANSFER', 'POS PURCHASE', 'SALARY', 'ATM WDL'])}. Avail Bal: GHS{amt()}",
    "g_promo_real": lambda: random.choice([
        f"Enjoy {random.choice(['50%', 'double'])} data bonus this weekend when you buy bundles via *138#. T&Cs apply.",
        "Pay for your ECG prepaid with MoMo and get instant tokens. Dial *170# > Pay Bill.",
        "Your MoMo statement is now available on the MoMo app. Download from Play Store or App Store.",
    ]),
    # Person-to-person genuine messages, including money talk.
    "g_personal_money": lambda: random.choice([
        f"Hi {random.choice(['bro', 'sis', 'mum', 'dad', 'chale'])}, I've sent the {random.choice(['money', 'cash', 'chop money'])}. Check your momo.",
        f"Please can you send me {cur()}{random.choice([20, 50, 100])} for transport? I'll pay you back on {random.choice(['Friday', 'month end', 'Monday'])}.",
        f"I sent you {cur()}{random.choice([50, 100, 200])} for the {random.choice(['rent', 'party', 'books', 'groceries'])}, confirm when you get it",
        "Mum the school fees have been paid, I'll send the receipt later",
        f"Did you receive the {cur()}{random.choice([30, 80, 150])}? Network was bad so I'm not sure it went",
        "Sorry I sent the money to the wrong number, I've called MTN to sort it out. Will resend tomorrow.",
    ]),
    "g_personal_other": lambda: random.choice([
        "Are we still meeting at 4? I'm at Circle already",
        "Happy birthday! God bless you, enjoy your day",
        "Please call me when you're free, it's about the wedding",
        "I'm on my way, traffic at Kaneshie is terrible",
        "Don't forget church tomorrow at 8",
        "The meeting has been moved to Thursday. Pass it on.",
        "Chale the match was something else yesterday",
    ]),
}

# ------------------------------------------------------------------- scam
SCAM = {
    # No "fake cash-in alert" template: a forged alert has the same text as a genuine one and
    # differs only in the sender (a personal number instead of the official ID). A text-only
    # model cannot tell them apart, so such a template is label noise. See README.
    #
    # Added in v2 after the first error analysis (pattern documented by Ghanaian sources:
    # "my correct number", "my wife's number"): asking for money to go to a *different* number.
    "s_other_number": lambda: random.choice([
        f"Hello, my line has a problem. Please send the money to this other number {phone()}, it will show {name()}",
        f"Good morning, don't use my old number. Transfer the amount to {phone()}, name is {name()}",
        f"Please the money you are sending, send it to my wife's number {phone()} instead",
        f"Send the {cur()}{amt(50, 800)} to this number {phone()}, my MoMo is blocked. It will come in the name {name()}",
        f"pls my momo get problem, push the money to {phone()} {name()}",
    ]),
    "s_by_mistake": lambda: random.choice([
        f"Please I mistakenly sent {cur()}{amt(50, 1000)} to your number. Kindly send it back to {phone()}. God bless you",
        f"Hello, I sent {cur()}{amt(50, 1000)} to you by mistake. It's my school fees, please reverse it to {phone()}",
        f"Good day, my brother sent money to your momo by mistake. Please return it, my mother is in hospital. Call {phone()}",
        f"Dear customer an amount of {cur()}{amt(100, 1500)} was wrongly transferred into your wallet. Kindly refund to {phone()} within 30 minutes",
        f"pls i send {cur()}{amt(50, 900)} to u wrongly mistake. send it back to my number {phone()} pls",
    ]),
    "s_blocked": lambda: random.choice([
        f"Your MoMo account has been blocked due to a wrong transaction. Call {phone()} immediately to unblock.",
        f"MTN: Your mobile money wallet will be suspended in 24 hours. Verify your details by calling {phone()}",
        f"Dear customer, your account is temporarily frozen. To reactivate dial *170# and enter your PIN, then call {phone()}",
        f"ALERT: Your SIM will be deactivated today for failing to update your Ghana Card details. Call customer care on {phone()}",
    ]),
    "s_promo": lambda: random.choice([
        f"Congratulations! Your number has won {cur()}{amt(1000, 20000)} in the MTN {random.choice(['Yello', 'MoMo', 'Anniversary'])} promo. Call {phone()} to claim your prize.",
        f"You have been selected as a winner of {cur()}{amt(1000, 10000)} in the ongoing promotion. Pay processing fee of {cur()}{amt(20, 200)} to receive your prize. Call {phone()}",
        f"CONGRATS!! You won a {random.choice(['Samsung phone', 'Toyota car', 'fridge', 'laptop'])} from {random.choice(['MTN', 'Telecel', 'Coca-Cola'])}. Send {cur()}{amt(50, 300)} delivery fee to {phone()}",
        f"Telecel promo: you are our lucky winner of {cur()}{amt(500, 5000)}. Claim now at {random.choice(['bit.ly/', 'telecel-promo.xyz/'])}{random.randint(100, 999)}",
    ]),
    "s_pin_otp": lambda: random.choice([
        f"MTN customer care: to complete your reversal please send the code you just received to {phone()}",
        f"We are updating our system. Reply with your MoMo PIN to avoid losing your money.",
        f"A code has been sent to your phone by mistake. Please send me the code, it is for my account. {phone()}",
        f"Dear customer to receive your bonus of {cur()}{amt(100, 1000)} dial *170# and enter your PIN to confirm",
    ]),
    "s_fake_agent": lambda: random.choice([
        f"Hello this is MoMo agent. Your cash out is pending, dial *170# select option 6 and approve to complete",
        f"This is the MTN head office. There is a problem with your account, call {phone()} now before it is closed",
        f"Mobile money security team: we noticed fraud on your wallet. Share the OTP sent to you to secure your money",
    ]),
    "s_data_bundle": lambda: f"{random.choice(['MTN', 'Telecel', 'AT'])} special offer: {random.choice(['20GB', '50GB', 'unlimited'])} data for only {cur()}{amt(5, 30)}. Send money to {phone()} {random.choice(['now', 'today only', 'offer ends tonight'])}",
    "s_loan_job": lambda: random.choice([
        f"Get instant loan of {cur()}{amt(1000, 10000)} with no collateral. Pay registration fee of {cur()}{amt(20, 150)} to {phone()}",
        f"Job opportunity at {random.choice(['Newmont', 'Ghana Gas', 'COCOBOD', 'GRA'])}. Salary {cur()}{amt(3000, 8000)} monthly. Send {cur()}{amt(50, 300)} application fee to {phone()}",
        f"Your loan of {cur()}{amt(500, 5000)} has been approved. Click {random.choice(['bit.ly/', 'qwik-loan.site/'])}{random.randint(100, 999)} and enter your PIN to receive",
    ]),
}


def generate(n_per_template=60, seed=42):
    random.seed(seed)
    rows = []
    for label, bank in [(0, GENUINE), (1, SCAM)]:
        for tid, fn in bank.items():
            seen = set()
            for _ in range(n_per_template * 4):
                if len(seen) >= n_per_template:
                    break
                t = " ".join(fn().split())
                if t not in seen:
                    seen.add(t)
                    rows.append({"text": t, "label": label, "source": "synthetic_ghana", "campaign_id": tid,
                                 "lang": "en", "is_synthetic": 1})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.label.value_counts().to_dict(), df.campaign_id.nunique(), "templates ->", OUT)
