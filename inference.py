import json
import os
import pickle

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


# Load artifacts
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
THRESHOLD = config["best_threshold"]


def predict(payload):
    cleaned = payload.lower().strip()
    X = vectorizer.transform([cleaned])
    # Get probability of malicious class (index 1)
    prob = model.predict_proba(X)[0, 1]
    pred = 1 if prob >= THRESHOLD else 0
    return pred, prob


if __name__ == "__main__":
    print("Enter payloads (empty line to quit):")
    while True:
        payload = input("Payload: ").strip()
        if not payload:
            break
        pred, prob = predict(payload)
        label = "Malicious" if pred else "Benign"
        print(f"{payload:40} -> {label:10} (confidence: {prob:.3f})")
