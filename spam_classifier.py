"""
spam_classifier.py
===================
A production-ready Spam/Ham text classification pipeline built with
scikit-learn and pandas.

Pipeline overview
------------------
1. Load a labeled spam/ham dataset (CSV with 'text' and 'label' columns).
   Falls back to a small built-in dummy dataset if no CSV is found, so the
   script always runs end-to-end out of the box.
2. Clean and preprocess the raw text (lowercasing, punctuation/number
   stripping, whitespace normalization).
3. Vectorize text with TF-IDF (unigrams + bigrams, English stop words removed).
4. Train a classifier (Multinomial Naive Bayes by default; Logistic
   Regression is also supported) inside a single sklearn Pipeline so that
   vectorizer + model always travel together.
5. Evaluate on a held-out test set: accuracy, precision, recall, F1,
   confusion matrix, and a full classification report.
6. Persist the trained pipeline to disk with joblib.
7. Expose `predict_message()` — a reusable function that takes any raw
   string and returns a ("Spam"/"Ham", confidence) tuple. This same
   function (and the saved model file) is what the Streamlit app uses.

Usage
-----
    python train.py                     # train + evaluate + save model
    python train.py --model logreg      # use Logistic Regression instead of NB
    python train.py --data path/to.csv  # use your own dataset

Note: training is run via the separate `train.py` entry point (rather than
`python spam_classifier.py` directly) so that this module is always
imported the same way — as `spam_classifier` — whether training from the
CLI or loading the cached model from the Streamlit app. This keeps the
pickled model's internal function references consistent between the two
entry points (a model trained under `__main__` cannot be safely unpickled
from a different importing module).

Dataset format expected
------------------------
A CSV with at least two columns:
    - 'text'  : the raw message/email content
    - 'label' : 'spam' or 'ham' (case-insensitive)

This matches common datasets such as the UCI/Kaggle "SMS Spam Collection"
and the "spam_ham_dataset.csv" enron-style dataset.
"""

import argparse
import os
import re
import string
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spam_ham_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.joblib")
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Data loading (with a built-in dummy fallback dataset)
# --------------------------------------------------------------------------

def _dummy_dataset() -> pd.DataFrame:
    """A tiny built-in dataset so the script is always runnable, even with
    no CSV file available (e.g. for a quick demo or a unit test)."""
    data = {
        "text": [
            "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
            "URGENT: Your account has been suspended. Verify your details immediately.",
            "Free entry into our $5000 prize draw, text WIN to 80085 now!",
            "You have been selected for a free cruise! Call now to claim your prize.",
            "Hot singles in your area want to meet you tonight, click now!",
            "Limited time offer: get 90% off Viagra, no prescription needed!",
            "Hey, are we still on for lunch tomorrow at noon?",
            "Can you send me the quarterly report before end of day?",
            "Don't forget to pick up milk on your way home.",
            "The meeting has been rescheduled to 3 PM on Thursday.",
            "Happy birthday! Hope you have a wonderful day.",
            "Please review the attached invoice and let me know if it looks correct.",
            "Reminder: your dentist appointment is tomorrow at 10 AM.",
            "Thanks for your help earlier, really appreciate it.",
            "Click this link to reset your bank password urgently or your account will be locked.",
            "Win a brand new iPhone by clicking this link right now, limited spots available!",
        ],
        "label": [
            "spam", "spam", "spam", "spam", "spam", "spam",
            "ham", "ham", "ham", "ham", "ham", "ham", "ham", "ham",
            "spam", "spam",
        ],
    }
    return pd.DataFrame(data)


def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a spam/ham dataset from CSV. Falls back to a dummy in-memory
    dataset if the file cannot be found or read, so the script never
    crashes just because a data file is missing.

    Expected columns: 'text' and 'label'. Common alternate column names
    (e.g. 'v1'/'v2' from the classic SMSSpamCollection.csv, or 'message')
    are auto-detected and renamed.
    """
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Could not read '{path}' ({exc}). Using dummy dataset instead.")
            return _dummy_dataset()
    else:
        print(f"[WARN] Dataset not found at '{path}'. Using a small built-in dummy dataset.")
        return _dummy_dataset()

    # Normalize column names across common dataset variants.
    col_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    # SMS Spam Collection classic format: v1=label, v2=text
    if "v1" in lower_cols and "v2" in lower_cols:
        col_map[lower_cols["v1"]] = "label"
        col_map[lower_cols["v2"]] = "text"
    else:
        if "label" in lower_cols:
            col_map[lower_cols["label"]] = "label"
        if "text" in lower_cols:
            col_map[lower_cols["text"]] = "text"
        elif "message" in lower_cols:
            col_map[lower_cols["message"]] = "text"
        elif "body" in lower_cols:
            col_map[lower_cols["body"]] = "text"

    df = df.rename(columns=col_map)

    if "label" not in df.columns or "text" not in df.columns:
        print("[WARN] Could not find 'label'/'text' columns. Using dummy dataset instead.")
        return _dummy_dataset()

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    df["text"] = df["text"].astype(str)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------------------------------

_SUBJECT_PREFIX_RE = re.compile(r"^\s*subject\s*:\s*", flags=re.IGNORECASE)
_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NUM_RE = re.compile(r"\b\d+\b")
_MULTISPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Lowercase and normalize a raw message string.

    Steps: strip a leading 'Subject:' prefix (common in email datasets),
    replace URLs/emails with placeholder tokens (their presence is a
    strong spam signal, but the exact address is not), strip standalone
    numbers and punctuation, and collapse whitespace.
    """
    text = text or ""
    text = _SUBJECT_PREFIX_RE.sub("", text)
    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = text.translate(_PUNCT_TABLE)
    text = _NUM_RE.sub(" numtoken ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# 3. Model building
# --------------------------------------------------------------------------

def build_pipeline(model_name: str = "naive_bayes") -> Pipeline:
    """Build a single sklearn Pipeline combining TF-IDF vectorization with
    a classifier, so preprocessing and the model are always saved/loaded
    together (avoids train/serve skew).
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    if model_name == "logreg":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=10.0,
            random_state=RANDOM_STATE,
        )
    elif model_name == "naive_bayes":
        classifier = MultinomialNB(alpha=0.3)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Use 'naive_bayes' or 'logreg'.")

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


# --------------------------------------------------------------------------
# 4. Training + evaluation
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    report: str


def train_and_evaluate(df: pd.DataFrame, model_name: str = "naive_bayes") -> tuple[Pipeline, EvalResult]:
    """Split the data, train the pipeline, and compute evaluation metrics
    on a held-out test set. 'spam' is treated as the positive class for
    precision/recall.
    """
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    result = EvalResult(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, pos_label="spam", zero_division=0),
        recall=recall_score(y_test, y_pred, pos_label="spam", zero_division=0),
        f1=f1_score(y_test, y_pred, pos_label="spam", zero_division=0),
        confusion=confusion_matrix(y_test, y_pred, labels=["ham", "spam"]),
        report=classification_report(y_test, y_pred, zero_division=0),
    )
    return pipeline, result


def print_evaluation(result: EvalResult) -> None:
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy:  {result.accuracy:.4f}")
    print(f"Precision (spam): {result.precision:.4f}")
    print(f"Recall (spam):    {result.recall:.4f}")
    print(f"F1-score (spam):  {result.f1:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted) [ham, spam]:")
    print(result.confusion)
    print("\nFull classification report:")
    print(result.report)
    print("=" * 60)


# --------------------------------------------------------------------------
# 5. Reusable prediction function
# --------------------------------------------------------------------------

def predict_message(message: str, pipeline: Pipeline | None = None) -> dict:
    """Classify a single raw message string as Spam or Ham.

    Parameters
    ----------
    message : str
        The raw email/SMS text to classify.
    pipeline : Pipeline, optional
        A trained sklearn Pipeline. If None, the pipeline is loaded from
        MODEL_PATH on disk (training one first if it doesn't exist yet).

    Returns
    -------
    dict with keys:
        'label'        : "Spam" or "Ham"
        'confidence'   : float in [0, 1], confidence in the predicted label
        'spam_probability' : float in [0, 1], raw P(spam)
        'ham_probability'  : float in [0, 1], raw P(ham)
    """
    if pipeline is None:
        pipeline = load_or_train_pipeline()

    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")

    classes = list(pipeline.named_steps["clf"].classes_)
    proba = pipeline.predict_proba([message])[0]
    spam_idx = classes.index("spam")
    ham_idx = classes.index("ham")

    spam_prob = float(proba[spam_idx])
    ham_prob = float(proba[ham_idx])
    label = "Spam" if spam_prob >= ham_prob else "Ham"
    confidence = max(spam_prob, ham_prob)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4),
    }


# --------------------------------------------------------------------------
# 6. Persistence helpers (shared with the Streamlit app)
# --------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    joblib.dump(pipeline, path)
    print(f"[INFO] Model saved to '{path}'")


def load_pipeline(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def load_or_train_pipeline(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = MODEL_PATH,
    model_name: str = "naive_bayes",
) -> Pipeline:
    """Convenience helper used by the Streamlit app: load a cached model
    from disk if present, otherwise train one from scratch and cache it.
    """
    if os.path.exists(model_path):
        try:
            return load_pipeline(model_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to load cached model ({exc}). Retraining...")

    df = load_dataset(data_path)
    pipeline, result = train_and_evaluate(df, model_name=model_name)
    print_evaluation(result)
    save_pipeline(pipeline, model_path)
    return pipeline


# --------------------------------------------------------------------------
# 7. CLI entry point
# --------------------------------------------------------------------------

def main():
    """CLI entry point. Invoked via train.py (see module docstring for why)."""
    parser = argparse.ArgumentParser(description="Train and evaluate a Spam/Ham text classifier.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV with 'text'/'label' columns.")
    parser.add_argument("--model", default="naive_bayes", choices=["naive_bayes", "logreg"],
                         help="Which classifier to train.")
    parser.add_argument("--out", default=MODEL_PATH, help="Where to save the trained model (joblib file).")
    args = parser.parse_args()

    print(f"[INFO] Loading dataset from '{args.data}' ...")
    df = load_dataset(args.data)
    print(f"[INFO] Loaded {len(df)} labeled messages "
          f"({(df['label'] == 'spam').sum()} spam / {(df['label'] == 'ham').sum()} ham).")

    print(f"[INFO] Training '{args.model}' classifier ...")
    pipeline, result = train_and_evaluate(df, model_name=args.model)
    print_evaluation(result)

    save_pipeline(pipeline, args.out)

    # Quick smoke-test of the reusable prediction function
    demo_messages = [
        "Congratulations! You have won a free lottery ticket, click here to claim your prize now!",
        "Hey, can we move our meeting to 4pm today?",
    ]
    print("\nSample predictions:")
    for msg in demo_messages:
        result = predict_message(msg, pipeline)
        print(f"  '{msg[:60]}...' -> {result['label']} "
              f"(confidence={result['confidence']:.2%}, "
              f"spam_prob={result['spam_probability']:.2%})")
