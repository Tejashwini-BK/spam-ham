"""
retrain_with_feedback.py
========================
Teach the spam detector from the corrections you saved in the app.

    python retrain_with_feedback.py

What it does:
  1. Reads feedback.csv (created when you press "This was wrong" in the app).
  2. Adds those corrected messages to the original training data.
  3. Trains a fresh model and compares it with your current one.
  4. Backs up your current model to spam_model.backup.joblib, then saves the new one
     as spam_model.joblib. (Use --dry-run to compare without saving.)

Then, in the app, press "Reload model" in the sidebar.
"""
import argparse
import os
import shutil
import warnings

warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

import spam_classifier as sc

HERE = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_PATH = os.path.join(HERE, "feedback.csv")
BACKUP_PATH = os.path.join(HERE, "spam_model.backup.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=8,
                        help="How many times each correction is repeated in training, so a few "
                             "corrections are not drowned out by thousands of old emails (default 8).")
    parser.add_argument("--dry-run", action="store_true", help="Compare models but do not save.")
    args = parser.parse_args()

    if not os.path.exists(FEEDBACK_PATH):
        print("No feedback.csv yet. Use the app and press 'This was wrong' on some mistakes first.")
        return
    fb = pd.read_csv(FEEDBACK_PATH).dropna(subset=["text", "label"])
    fb["label"] = fb["label"].astype(str).str.strip().str.lower()
    fb = fb[fb["label"].isin(["spam", "ham"])].drop_duplicates(subset=["text"])
    if fb.empty:
        print("feedback.csv has no usable rows.")
        return
    print(f"[INFO] Found {len(fb)} saved corrections "
          f"({(fb.label == 'spam').sum()} spam, {(fb.label == 'ham').sum()} ham).")

    df = sc.load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=sc.RANDOM_STATE, stratify=df["label"])

    X_aug = pd.concat([X_train, pd.concat([fb["text"]] * args.repeat)], ignore_index=True)
    y_aug = pd.concat([y_train, pd.concat([fb["label"]] * args.repeat)], ignore_index=True)
    new = sc.build_pipeline()
    new.fit(X_aug, y_aug)

    def report(name, model):
        acc = accuracy_score(y_test, model.predict(X_test))
        fixed = (model.predict(fb["text"]) == fb["label"]).mean()
        print(f"  {name:<12} accuracy on original test emails: {acc:.1%} | your corrections now right: {fixed:.0%}")

    print("[INFO] Comparison:")
    if os.path.exists(sc.MODEL_PATH):
        report("current", joblib.load(sc.MODEL_PATH))
    report("retrained", new)

    if args.dry_run:
        print("[INFO] Dry run: nothing saved.")
        return
    if os.path.exists(sc.MODEL_PATH):
        shutil.copy2(sc.MODEL_PATH, BACKUP_PATH)
        print(f"[INFO] Old model backed up to '{os.path.basename(BACKUP_PATH)}'")
    joblib.dump(new, sc.MODEL_PATH)
    print(f"[INFO] New model saved to '{os.path.basename(sc.MODEL_PATH)}'. In the app, press 'Reload model'.")


if __name__ == "__main__":
    main()
