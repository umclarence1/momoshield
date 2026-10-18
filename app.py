"""Streamlit demo.  Run:  streamlit run app.py"""
import html
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predictor import Predictor  # noqa: E402

st.set_page_config(page_title="MoMo Scam Check", page_icon="📱")


@st.cache_resource
def load():
    return Predictor()


predictor = load()

st.title("📱 MoMo Scam Check")
st.caption("Paste an SMS to check it for common Mobile Money scam patterns. Check before you send.")

EXAMPLES = {
    "Mistaken transfer": "You have received GH₵500 by mistake. Send it back to 0244123456 immediately or your account will be blocked.",
    "Genuine alert": "Payment received for GHS 120.00 from AKUA DARKO Current Balance: GHS 340.55. Reference: lunch. Transaction ID: 51234098765. TRANSACTION FEE: 0.00",
    "Fake promo": "Congratulations! Your number has won GHS 5,000 in the MoMo promo. Pay GHS 50 processing fee to 0551234567 to receive your prize.",
}
choice = st.selectbox("Try an example", ["—"] + list(EXAMPLES))
text = st.text_area("SMS text", value=EXAMPLES.get(choice, ""), height=120)

if st.button("Check message", type="primary") and text.strip():
    r = predictor.analyse(text)
    colour = {"high": "🔴", "medium": "🟠", "low": "🟢"}[r["level"]]
    label = {"high": "Likely scam", "medium": "Suspicious", "low": "No common scam patterns found"}[r["level"]]
    st.subheader(f"{colour} {label}  ·  risk {r['risk']:.0%}")

    if r["reasons"]:
        st.markdown("**Reasons for concern**")
        for reason in r["reasons"]:
            st.markdown(f"- {reason}")

    st.markdown("**What you should do**")
    for a in r["advice"]:
        st.markdown(f"- {a}")

    if r["influential_words"]:
        shown = html.escape(text)
        for w in sorted(r["influential_words"], key=len, reverse=True):
            if w.startswith("<"):
                continue
            shown = re.sub(rf"(?i)\b({re.escape(html.escape(w))})\b",
                           r"<mark style='background:#ffb3b3'>\1</mark>", shown)
        with st.expander("Words the model weighed most"):
            st.markdown(shown, unsafe_allow_html=True)
            st.caption("Highlighted words pushed the model towards 'scam'. They show how the model decided, not proof of fraud.")

st.divider()
st.caption("This is a research prototype. A warning is not proof of fraud, and a clean result does not mean a message is safe. "
           "Always check your balance through the official MoMo menu before sending money.")
