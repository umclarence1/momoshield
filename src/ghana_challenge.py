"""Hand-written Ghana challenge set (evaluation only).

These are CONSTRUCTED messages, written from scam patterns described by
Ghanaian news, telco and Cyber Security Authority warnings. They are not
real received messages, so results on this set are a stress test, not
evidence of real-world accuracy. They were written separately from the
synthetic_ghana.py templates, with different wording, and include:
  - hard negatives: genuine messages full of words like "reversal", "PIN", "urgent"
  - evasive scams: no "mistake", no "prize", no links
Columns: text, label (1 = scam), kind.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "challenge" / "ghana_challenge.csv"

ROWS = [
    # ---- obvious scams
    ("Hi I'm the one who sent you GHS 300 just now, it was meant for my sister. Abeg return am to 0551234567", 1, "scam_mistake"),
    ("Sorry to disturb you, I made an error and the money went to you. Kindly help me and send to my other line 0209876543", 1, "scam_mistake"),
    ("Cash In received for GHS 750.00 from ESI NYARKO. Your balance is GHS 752.10. please confirm receipt", 1, "scam_fake_alert"),
    ("You have received GHS1,200.00 from YAW BOAKYE. New balance GHS1,204.00", 1, "scam_fake_alert"),
    ("Your Mobile Money account is under review. Failure to call 0598765432 in the next hour will lead to closure.", 1, "scam_threat"),
    ("Final notice: your Telecel line will be barred tonight. Contact the agent on 0501112223 to keep your line active", 1, "scam_threat"),
    ("Good news! Your number was picked in the MoMo end of year draw. You get GHS 8,000. Call 0245556667 for details", 1, "scam_promo"),
    ("You are entitled to a GHS 2,500 COVID relief payment from government. Pay GHS 50 clearance to 0573332221", 1, "scam_promo"),
    ("Dear user your bonus is ready. Dial *170# choose Approve and put your PIN to collect it", 1, "scam_ussd"),
    ("I am calling from MTN, a reversal code will come to your phone, read it to me so we can cancel the wrong payment", 1, "scam_otp"),
    ("Ghana Card linking: your SIM is not registered. Click ghana-sim-verify.com and enter your details to avoid disconnection", 1, "scam_link"),
    ("Hello dear, I am stranded at Kumasi station, my phone was stolen. This is my new number. Send GHS 200 for transport please", 1, "scam_impersonate"),
    ("Mum it's me, I lost my phone. Please send 500 cedis to this number, I'll explain later", 1, "scam_impersonate"),
    ("Earn GHS 300 daily from home liking YouTube videos. Register with GHS 100 via momo to 0244445556", 1, "scam_job"),
    ("Invest GHS 500 today and get GHS 2,000 back in 7 days. Many people have already benefited. WhatsApp 0267778889", 1, "scam_invest"),
    # ---- evasive scams (no classic trigger words)
    ("Hello, did you see the money I sent? Please check and call me back on this number", 1, "scam_evasive"),
    ("Boss, the GHS 450 was for the supplier, not you. Do the needful and push it to 0554443332", 1, "scam_evasive"),
    ("Your transaction of GHS 980 is on hold. Our officer will call you shortly, follow his instructions", 1, "scam_evasive"),
    ("Charley the momo I sent, the person says it hasn't arrived and it shows on your side. Help me out, send it to him direct", 1, "scam_evasive"),
    ("This is the Accra Central MoMo office. Kindly pick our call for an important update on your wallet", 1, "scam_evasive"),
    # ---- genuine alerts
    ("Payment received for GHS 120.00 from AKUA DARKO Current Balance: GHS 340.55 Available Balance: GHS 340.55. Reference: lunch. Transaction ID: 51234098765. TRANSACTION FEE: 0.00", 0, "genuine_alert"),
    ("Payment made for GHS 60.00 to KWESI ARTHUR. Current Balance: GHS 22.10. Reference: 1. Transaction ID: 51230987123. Fee charged: GHS 0.60", 0, "genuine_alert"),
    ("Telecel Cash: You have received GHS 45.00 from NANA ADJEI. Your new balance is GHS 63.20. Transaction ID: 001234567890.", 0, "genuine_alert"),
    ("Your ECG prepaid purchase of GHS 100.00 was successful. Token: 1234 5678 9012 3456 7890. Thank you.", 0, "genuine_alert"),
    # ---- hard negatives (genuine, but scary-sounding)
    ("Your transaction of GHS 200.00 to the wrong number has been reversed following your complaint. Ref 51209876543.", 0, "hard_negative"),
    ("MTN: We will never ask for your MoMo PIN. If anyone calls claiming you received money by mistake, report to 1515.", 0, "hard_negative"),
    ("Your Bolt verification code is 482913. Do not share this code with anyone.", 0, "hard_negative"),
    ("Urgent: the church meeting has been moved to 6pm today. Please inform the others immediately.", 0, "hard_negative"),
    ("Kofi I mistakenly sent you 50 instead of 30, just keep the 20 for credit", 0, "hard_negative"),
    ("Please send back the book I borrowed you when you're coming tomorrow", 0, "hard_negative"),
    ("Dad please send GHS 100 for my exam registration, deadline is Friday", 0, "hard_negative"),
    ("Your account will be debited GHS 25.00 for your monthly DStv subscription on 05/11. Ensure sufficient balance.", 0, "hard_negative"),
    ("Congratulations on your graduation! So proud of you. Sending you a small gift on momo", 0, "hard_negative"),
    ("Your MoMo wallet has been temporarily locked after 3 wrong PIN attempts. Visit any MTN service centre with your Ghana Card.", 0, "hard_negative"),
    ("I'm at the agent's place, his network is down so I'll send the money later. Don't worry.", 0, "hard_negative"),
    ("The landlord says pay the rent by Monday or he'll lock the gate. Can you send your share?", 0, "hard_negative"),
    # ---- ordinary genuine
    ("Abeg buy me pure water on your way", 0, "genuine_personal"),
    ("Thank you for the help yesterday, God bless you", 0, "genuine_personal"),
    ("Reminder: your appointment at Korle Bu is on Tuesday at 9am", 0, "genuine_personal"),
    ("You have successfully renewed your 3GB monthly bundle. Expiry: 17/11/2026.", 0, "genuine_service"),
]


if __name__ == "__main__":
    df = pd.DataFrame(ROWS, columns=["text", "label", "kind"])
    df["source"] = "ghana_challenge_constructed"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.label.value_counts().to_dict(), "->", OUT)
