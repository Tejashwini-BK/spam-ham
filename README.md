# Spam & Ham Detector

A complete spam/ham text classification system: a reusable scikit-learn
pipeline (TF-IDF + Naive Bayes) plus a Streamlit web app for real-time use.

## Files

| File | Purpose |
|---|---|
| `spam_classifier.py` | Core library: data loading, text cleaning, TF-IDF + model pipeline, training, evaluation, and the reusable `predict_message()` function. |
| `train.py` | CLI entry point — run this to train and evaluate the model. |
| `app.py` | Streamlit web app for real-time classification in the browser. |
| `data/spam_ham_dataset.csv` | Your dataset (5,171 labeled emails). |
| `spam_model.joblib` | Saved trained pipeline (created after first training run). |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train & evaluate the model (CLI)

```bash
python train.py
```

This will:
- Load `data/spam_ham_dataset.csv` (auto-detects `text`/`label` columns; also handles the classic `v1`/`v2` SMS Spam Collection format)
- Clean text and vectorize with TF-IDF (unigrams + bigrams)
- Train a Multinomial Naive Bayes classifier
- Print Accuracy, Precision, Recall, F1, and a Confusion Matrix
- Save the trained pipeline to `spam_model.joblib`

Optional flags:

```bash
python train.py --model logreg              # use Logistic Regression instead
python train.py --data path/to/other.csv    # train on a different dataset
```

### Use the prediction function in your own code

```python
from spam_classifier import predict_message

result = predict_message("Congratulations! You've won a free prize, click now!")
print(result)
# {'label': 'Spam', 'confidence': 0.97, 'spam_probability': 0.97, 'ham_probability': 0.03}
```

## 2. Run the web app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`). Paste
any message into the text box and click **Analyze Message** to see the
Spam/Ham label, confidence, and a probability breakdown chart. The first
run will auto-train the model if `spam_model.joblib` doesn't exist yet;
subsequent runs load the cached model instantly.

## Notes

- Both entry points share the same `spam_classifier.py` module, so the
  model trained via `train.py` is guaranteed to load correctly in `app.py`
  (see the comment in `train.py` for why this matters).
- `class_weight="balanced"` / `alpha` tuning and bigram features are used
  to handle the moderate class imbalance in the dataset (~71% ham / 29% spam).
- To swap in a different dataset (e.g. the classic SMS Spam Collection),
  just point `--data` at a CSV with `text`/`label` (or `v1`/`v2`) columns.
