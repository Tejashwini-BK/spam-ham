"""
app.py
======
A lightweight Streamlit web application for real-time Spam/Ham detection.

Run locally with:
    streamlit run app.py

On first run, if no cached model file (spam_model.joblib) is found, the app
will automatically train one from data/spam_ham_dataset.csv (or a small
built-in dummy dataset if that file is missing) and cache it for future runs.
"""

import csv
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from spam_classifier import (
    DEFAULT_DATA_PATH,
    MODEL_PATH,
    load_or_train_pipeline,
    predict_message,
)

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

# --------------------------------------------------------------------------
# Model loading (cached across reruns so the app stays fast)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load the trained pipeline, training it once if necessary.
    Cached with st.cache_resource so this only runs once per server session,
    not on every user interaction / rerun.
    """
    return load_or_train_pipeline(data_path=DEFAULT_DATA_PATH, model_path=MODEL_PATH)


# --------------------------------------------------------------------------
# Feedback ("this was wrong") storage
# --------------------------------------------------------------------------

FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")
FEEDBACK_COLUMNS = ["timestamp", "text", "label", "predicted", "spam_probability"]


def count_feedback() -> int:
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    try:
        return len(pd.read_csv(FEEDBACK_PATH))
    except Exception:
        return 0


def save_feedback(text: str, predicted: str, spam_probability: float) -> bool:
    """Save a correction. 'label' is the CORRECT answer (the opposite of what the
    model predicted), in the same 'spam'/'ham' format as the training data.
    Returns False if this exact message was already saved."""
    correct = "ham" if predicted.lower() == "spam" else "spam"
    if os.path.exists(FEEDBACK_PATH):
        try:
            if text in set(pd.read_csv(FEEDBACK_PATH)["text"].astype(str)):
                return False
        except Exception:
            pass
    is_new_file = not os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new_file:
            writer.writerow(FEEDBACK_COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), text, correct,
                         predicted.lower(), f"{spam_probability:.4f}"])
    return True


def clear_message():
    st.session_state["message_box"] = ""
    st.session_state.pop("last", None)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app uses **TF-IDF** vectorization with a **Naive Bayes**
        classifier (scikit-learn) to detect whether a message is
        **Spam** or **Ham** (legitimate) in real time.

        **How it works**
        1. Your text is cleaned and converted into TF-IDF features.
        2. A trained classifier scores the message.
        3. You get a label plus a confidence / probability breakdown.
        """
    )
    st.divider()
    st.caption("Model file: `spam_model.joblib` (auto-trained on first run)")
    st.divider()
    st.subheader("Your corrections")
    st.metric("Saved so far", count_feedback())
    st.caption(
        "Every time you press **This was wrong**, the message is saved to "
        "`feedback.csv` on your computer. Run `python retrain_with_feedback.py` "
        "to teach the model from them, then press the button below."
    )
    if st.button("🔄 Reload model", use_container_width=True):
        get_pipeline.clear()
        st.rerun()

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------

st.title("🛡️ Spam & Ham Detector")
st.write("Paste an email or text message below and get an instant classification.")

with st.spinner("Loading model... (first run may take a moment to train)"):
    pipeline = get_pipeline()

message = st.text_area(
    "Message to analyze",
    height=180,
    placeholder="Paste your email or SMS text here...",
    key="message_box",
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
with col2:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_message)

# --------------------------------------------------------------------------
# Example messages for quick testing
# --------------------------------------------------------------------------

with st.expander("✨ Try an example"):
    examples = {
        "Spam example": "Congratulations! You've won a $1,000 gift card. Click here now to claim your prize before it expires!",
        "Ham example": "Hi, just checking if we're still meeting tomorrow at 2pm to review the project timeline.",
    }
    ex_col1, ex_col2 = st.columns(2)
    if ex_col1.button(list(examples.keys())[0], use_container_width=True):
        message = examples["Spam example"]
        analyze_clicked = True
    if ex_col2.button(list(examples.keys())[1], use_container_width=True):
        message = examples["Ham example"]
        analyze_clicked = True

# --------------------------------------------------------------------------
# Run prediction and display results
# --------------------------------------------------------------------------

if analyze_clicked:
    if not message or not message.strip():
        st.warning("⚠️ Please enter a message to analyze.")
        st.session_state.pop("last", None)
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.2)  # brief pause so the spinner is visible for very fast predictions
            try:
                result = predict_message(message, pipeline)
            except Exception as exc:
                st.error(f"Something went wrong during prediction: {exc}")
                result = None
        if result:
            # Remember the result so it stays on screen when a feedback button is pressed
            st.session_state["last"] = {"message": message, "result": result, "saved": None}

last = st.session_state.get("last")
if last:
    result = last["result"]
    st.divider()

    if result["label"] == "Spam":
        st.error(f"🚨 **SPAM** detected — confidence: {result['confidence']:.1%}")
    else:
        st.success(f"✅ **HAM** (legitimate) — confidence: {result['confidence']:.1%}")

    st.subheader("Probability breakdown")
    prob_df = pd.DataFrame(
        {
            "Class": ["Ham", "Spam"],
            "Probability": [result["ham_probability"], result["spam_probability"]],
        }
    ).set_index("Class")
    st.bar_chart(prob_df)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Prediction", result["label"])
    metric_col2.metric("Spam probability", f"{result['spam_probability']:.1%}")
    metric_col3.metric("Ham probability", f"{result['ham_probability']:.1%}")

    # ---- "This was wrong" feedback ----
    st.markdown("**Was this answer wrong?**")
    actually = "HAM (legitimate)" if result["label"] == "Spam" else "SPAM"
    if last["saved"] is None:
        if st.button(f"👎 This was wrong — it's actually {actually}", key="wrong_btn"):
            last["saved"] = save_feedback(last["message"], result["label"], result["spam_probability"])
            st.rerun()
    elif last["saved"]:
        st.success("Thanks! Correction saved to `feedback.csv`. It will be used next time you retrain.")
    else:
        st.info("You already reported this exact message. Nothing new was saved.")

st.divider()
st.caption(
    "Built with scikit-learn (TF-IDF + Naive Bayes) and Streamlit. "
    "This tool is for demonstration purposes and should not be the sole "
    "safeguard against malicious messages."
)
