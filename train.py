import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Configuration
DATA_PATH = "clean_payloads.csv"
OUTPUT_DIR = "model"
VALIDATION_SIZE = 0.2  # (threshold tuning + final eval)
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Load & preprocess
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df.drop_duplicates(subset=["Payload"], keep="first", inplace=True)
df["Payload"] = df["Payload"].astype(str).str.lower().str.strip()
print(f"Total samples after dedup: {len(df)}")


# 2. Split into train and validation
print("Splitting data (stratified by 'Type')...")
train_df, val_df = train_test_split(
    df, test_size=VALIDATION_SIZE, stratify=df["Type"], random_state=RANDOM_SEED
)
print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

# Labels (binary)
y_train = (train_df["Type"] != "benign").astype(int)
y_val = (val_df["Type"] != "benign").astype(int)


# 3. Feature extraction
print("Extracting character n-gram features (2-5)...")
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=20000,
    lowercase=True,
    sublinear_tf=True,
)
X_train = vectorizer.fit_transform(train_df["Payload"])
X_val = vectorizer.transform(val_df["Payload"])
print(f"Feature matrix shape: {X_train.shape}")


# 4. Baseline (Dummy)
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_val, dummy.predict(X_val))
print(f"Baseline (dummy) accuracy: {dummy_acc:.4f}")


# 5. Hyperparameter tuning (C)
print("Tuning Logistic Regression (C parameter) ...")
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
lr_base = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    random_state=RANDOM_SEED,
)
grid = GridSearchCV(lr_base, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best_lr = grid.best_estimator_
print(f"Best C: {grid.best_params_['C']}")


# 6. Threshold tuning on validation set
prob_val = best_lr.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.1, 0.9, 50)
best_thr = 0.5
best_f1 = 0
for thr in thresholds:
    pred = (prob_val >= thr).astype(int)
    f1 = f1_score(y_val, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
print(f"Optimal threshold (on validation set): {best_thr:.2f} (F1 = {best_f1:.4f})")


# 7. Final evaluation on validation set
y_pred = (prob_val >= best_thr).astype(int)

metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "precision": precision_score(y_val, y_pred, zero_division=0),
    "recall": recall_score(y_val, y_pred, zero_division=0),
    "f1": f1_score(y_val, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_val, prob_val),
}
print(f"\nFinal validation metrics: {metrics}")

print("\n" + "=" * 60)
print(f"CLASSIFICATION REPORT (Threshold = {best_thr:.2f})")
print("=" * 60)
print(classification_report(y_val, y_pred, target_names=["Benign", "Malicious"]))

cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:\n", cm)


# 8. Save model, vectorizer, config
with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_lr, f)
with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

config = {
    "model": "LogisticRegression",
    "vectorizer": "TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_features=20000)",
    "best_params": grid.best_params_,
    "best_threshold": float(best_thr),
    "seed": RANDOM_SEED,
    "split_ratios": {"train": 0.8, "validation": 0.2},
    "metrics": metrics,
    "date": datetime.now().isoformat(),
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)


# 9. Figures
def save_figure(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["Benign", "Malicious"],
    yticklabels=["Benign", "Malicious"],
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Validation Set)")
save_figure(fig, "confusion_matrix.png")


# ROC curve
fpr, tpr, _ = roc_curve(y_val, prob_val)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, label=f"LR (AUC = {metrics['roc_auc']:.3f})", lw=2)
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
save_figure(fig, "roc_curve.png")


# Feature importance (top 20 coefficients)
coefs = best_lr.coef_[0]
feature_names = vectorizer.get_feature_names_out()
top_idx = np.argsort(np.abs(coefs))[-20:]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(top_idx)), coefs[top_idx])
ax.set_yticks(range(len(top_idx)))
ax.set_yticklabels([feature_names[i] for i in top_idx])
ax.set_xlabel("Logistic Regression Coefficient")
ax.set_title("Top 20 Most Influential Character n‑grams")
save_figure(fig, "feature_importance.png")

print("All done! Model, vectorizer, config, and figures saved.")
