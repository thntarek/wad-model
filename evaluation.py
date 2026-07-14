import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def load_artifacts(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    threshold = config.get("best_threshold", 0.5)
    return model, vectorizer, threshold, config


def load_test_data(csv_path):
    df = pd.read_csv(csv_path)
    if "Payload" in df.columns and "Type" in df.columns:
        X = df["Payload"].astype(str)
        y = (df["Type"] != "benign").astype(int)
    elif "payload" in df.columns and "label" in df.columns:
        X = df["payload"].astype(str)
        y = df["label"].astype(int)
    else:
        raise ValueError(
            "CSV must have columns: 'Payload' and 'Type' OR 'payload' and 'label'"
        )
    X_clean = X.str.lower().str.strip()
    return X_clean, y, df


def evaluate(model, vectorizer, X, y, threshold):
    X_vec = vectorizer.transform(X)
    probs = model.predict_proba(X_vec)[:, 1]
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
    }
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=["Benign", "Malicious"])
    return preds, probs, metrics, cm, report


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on the test set.")
    parser.add_argument(
        "--test_csv",
        type=str,
        default="model/test_data.csv",
        help="Path to test CSV (default: model/test_data.csv)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model",
        help="Directory containing model.pkl, vectorizer.pkl, config.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation", help="Directory to save outputs"
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_csv):
        print(f"Error: Test CSV not found at '{args.test_csv}'.")
        print(
            "Please run train.py first to generate the test set, or specify --test_csv."
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)

    model, vectorizer, threshold, config = load_artifacts(args.model_dir)
    print(f"Loaded model from {args.model_dir} (threshold = {threshold:.2f})")

    X, y, df = load_test_data(args.test_csv)
    print(f"Loaded test set with {len(df)} samples")

    preds, probs, metrics, cm, report = evaluate(model, vectorizer, X, y, threshold)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS (threshold = {threshold:.2f})")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:10} : {v:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    df_out = df.copy()
    df_out["predicted_label"] = preds
    df_out["probability"] = probs
    pred_csv = os.path.join(args.output_dir, "test_predictions.csv")
    df_out.to_csv(pred_csv, index=False)
    print(f"\nPredictions saved to {pred_csv}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output_dir}/metrics.json")

    plot_confusion_matrix(
        cm, save_path=os.path.join(args.output_dir, "confusion_matrix.png")
    )
    plot_roc_curve(y, probs, save_path=os.path.join(args.output_dir, "roc_curve.png"))

    print("\nAll done! Outputs saved in '{}'.".format(args.output_dir))


if __name__ == "__main__":
    main()
