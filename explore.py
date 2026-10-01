"""Exploration helper (does NOT modify the project's own files)."""
import warnings; warnings.filterwarnings("ignore")
import sys, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import spam_classifier as sc

df = sc.load_dataset()
X, y = df["text"], df["label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=sc.RANDOM_STATE, stratify=y)
old = joblib.load("spam_model.joblib")

# ---- Part A: the mistakes on the held-out test emails ----
pred = old.predict(Xte); prob = old.predict_proba(Xte)[:, list(old.classes_).index("spam")]
wrong = Xte[pred != yte]
print("=== A. MISTAKES ===")
for kind, actual in (("Normal mail wrongly flagged as SPAM", "ham"), ("SPAM that slipped through", "spam")):
    idx = [i for i in wrong.index if yte[i] == actual][:3]
    print(f"\n{kind}:")
    for i in idx:
        snippet = " ".join(Xte[i].split())[:130]
        print(f"  - ({prob[list(Xte.index).index(i)]*100:.0f}% spam) {snippet}")

# ---- Part B/C: modern messages the model never saw ----
modern_test = [
 ("URGENT: your account is suspended, verify your password here", "spam"),
 ("Your package could not be delivered. Confirm your address and pay a small fee to reschedule", "spam"),
 ("You have been selected for a $1000 gift card, claim today", "spam"),
 ("Unusual sign-in detected on your bank account. Log in now to secure it or it will be locked", "spam"),
 ("Earn $500 a day from home with this crypto trading secret, limited spots", "spam"),
 ("Your subscription payment failed. Update your billing details immediately to avoid cancellation", "spam"),
 ("Hey, are we still on for lunch tomorrow?", "ham"),
 ("Can you send me the Q3 report by Friday?", "ham"),
 ("Thanks for the update, I will review the draft tonight and send comments", "ham"),
 ("Reminder: team standup moved to 10:30 tomorrow, same video link", "ham"),
 ("Happy to jump on a call Thursday afternoon to go over the budget", "ham"),
 ("Attached is the signed contract, let me know if anything looks off", "ham"),
]
extra_spam = [
 "Your account has been suspended. Verify your identity now to restore access",
 "Security alert: someone tried to sign in to your account. Confirm your password immediately",
 "Your parcel is on hold. Pay the delivery fee here to release it",
 "Congratulations, you are our lucky winner of a gift card. Click to claim your reward now",
 "Final notice: your payment method was declined, update billing information today or lose access",
 "Verify your bank login now to avoid account closure",
 "Claim your tax refund today, submit your bank details on this secure link",
 "Make money fast with crypto, guaranteed returns, join now",
 "Your password expires today. Click here to keep your account active",
 "You've won a free vacation. Confirm your details to receive your prize",
 "Unusual activity detected. Login here to unlock your account immediately",
 "Limited time offer: get a free gift card by completing this short survey",
]
extra_ham = [
 "Are you free for coffee this afternoon?",
 "Please find the meeting notes attached, let me know your thoughts",
 "Can we move our call to Wednesday morning?",
 "Thanks for sending the invoice, I will get it approved this week",
 "Reminder: the project deadline is next Friday",
 "I reviewed your proposal and have a few questions about the timeline",
 "Dinner at 7 on Saturday? Let me know if that works",
 "Here is the updated schedule for next week's workshop",
 "Great job on the presentation today, the client was really happy",
 "Could you share the slides before the meeting tomorrow?",
 "Just checking in to see how the onboarding is going",
 "The package arrived yesterday, thanks so much for sending it",
]
def score(model, X_, y_):
    p = model.predict(X_)
    return accuracy_score(y_, p), recall_score(y_, p, pos_label="spam"), precision_score(y_, p, pos_label="spam")

REPEAT = 8   # repeat the small hand-written set so it is not drowned out by 4,000 old emails
Xaug = pd.concat([Xtr, pd.Series((extra_spam + extra_ham) * REPEAT)], ignore_index=True)
yaug = pd.concat([ytr, pd.Series((["spam"] * len(extra_spam) + ["ham"] * len(extra_ham)) * REPEAT)], ignore_index=True)
new = sc.build_pipeline(); new.fit(Xaug, yaug)
joblib.dump(new, "spam_model_v2.joblib")

mt, my = pd.Series([m for m, _ in modern_test]), pd.Series([l for _, l in modern_test])
print("\n=== B. MODERN MESSAGES: old model vs new model ===")
op, np_ = old.predict(mt), new.predict(mt)
for m, l, a, b in zip(mt, my, op, np_):
    print(f"MODERN|{l}|{a}|{b}|{m}")
print("\n=== C. SCOREBOARD ===")
for name, mdl in (("old", old), ("new", new)):
    a1, r1, p1 = score(mdl, Xte, yte); a2, r2, p2 = score(mdl, mt, my)
    print(f"SCORE|{name}|old-style emails accuracy {a1*100:.1f}%|modern accuracy {a2*100:.0f}%|modern spam caught {r2*100:.0f}%")

# ---- interactive helper: python explore.py "your message" ----
for msg in sys.argv[1:]:
    for name, mdl in (("old", old), ("new", new)):
        p = mdl.predict_proba([msg])[0][list(mdl.classes_).index("spam")]
        print(f"YOURS|{name}|{'Spam' if p>0.5 else 'Ham'}|{p*100:.0f}%|{msg}")
