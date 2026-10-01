"""
train.py
========
CLI entry point for training the spam classifier.

Run this instead of executing spam_classifier.py directly:

    python train.py
    python train.py --model logreg
    python train.py --data path/to/your_dataset.csv

Why a separate entry point?
----------------------------
`spam_classifier.py` is designed to be imported as a normal module both
here and from `app.py`. If you ran `python spam_classifier.py` directly,
Python would load it under the special module name `__main__`, and the
trained model's TF-IDF preprocessor (a function defined in that module)
would be pickled with a reference to `__main__`. When `app.py` later tries
to import `spam_classifier` normally and load that cached model, unpickling
would fail because `__main__.clean_text` doesn't exist in that context.

Routing training through this tiny wrapper ensures `spam_classifier` is
always imported the same way, so the saved model works everywhere.
"""

from spam_classifier import main

if __name__ == "__main__":
    main()
