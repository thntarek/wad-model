import json, os, pickle, sys
import torch, torch.nn as nn, torch.nn.functional as F

MODEL_DIR = "model"  

class CharTokenizer:
    PAD, UNK = "<PAD>", "<UNK>"
    def __init__(self, max_len):
        self.max_len = max_len
        self.char2idx = {self.PAD: 0, self.UNK: 1}
    def fit(self, texts):
        for c in sorted(set("".join(texts))):
            self.char2idx.setdefault(c, len(self.char2idx))
    @property
    def vocab_size(self):
        return len(self.char2idx)
    def encode(self, text):
        ids = [self.char2idx.get(c, 1) for c in str(text)[: self.max_len]]
        return ids + [0] * (self.max_len - len(ids))
    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.v = nn.Parameter(torch.randn(dim) * 0.1)
    def forward(self, x, mask):
        u = torch.tanh(self.proj(x))
        scores = torch.matmul(u, self.v).masked_fill(mask == 0, -1e9)
        w = F.softmax(scores, dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1), w

class PayloadClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, filters=64, hidden=128,
                 dense=128, dropout=0.5, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv3 = nn.Conv1d(embed_dim, filters, 3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, filters, 5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, filters, 7, padding=3)
        self.lstm = nn.LSTM(filters * 3, hidden, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden * 2)
        self.fc1 = nn.Linear(hidden * 2, dense)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense, 1)
    def forward(self, x):
        mask = (x != self.pad_idx).float()
        e = self.embed(x).permute(0, 2, 1)
        c = torch.cat([F.relu(self.conv3(e)), F.relu(self.conv5(e)), F.relu(self.conv7(e))], dim=1)
        o, _ = self.lstm(c.permute(0, 2, 1))
        ctx, _ = self.attn(o, mask)
        h = self.drop(F.relu(self.fc1(ctx)))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


CLASS_NAMES = {0: "Benign", 1: "Malicious"}


def load_artifacts():
    for name in ("best_model.pth", "tokenizer.pkl", "config.json"):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' not found. Run train.py first to create it.")

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tok = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PayloadClassifier(
        vocab_size=config["vocab_size"], embed_dim=config["embed_dim"],
        filters=config["cnn_filters"], hidden=config["lstm_hidden"],
        dense=config["dense_dim"], dropout=config["dropout"], pad_idx=config["pad_idx"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=device))
    model.eval()
    return model, tok, device


def predict(payload: str, model, tok, device, threshold: float = 0.5):
    x = torch.tensor([tok.encode(payload)], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = model(x).item()
    return prob, int(prob >= threshold)


def main():
    print("[INFO] Loading model...")
    model, tok, device = load_artifacts()
    print(f"[INFO] Ready. Device: {device}\n")

    # Single-shot mode: payload passed as a command-line argument
    if len(sys.argv) > 1:
        payload = " ".join(sys.argv[1:])
        prob, label = predict(payload, model, tok, device)
        print(f"Payload : {payload}")
        print(f"Prob(1) : {prob:.4f}")
        print(f"Label   : {label} ({CLASS_NAMES[label]})")
        return

    # Interactive mode
    print("Enter a payload to classify ('exit' to quit):")
    while True:
        try:
            payload = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break
        if payload.lower() in ("exit", "quit", ""):
            print("[INFO] Exiting.")
            break
        prob, label = predict(payload, model, tok, device)
        print(f"    Probability (Malicious): {prob:.4f}")
        print(f"    Predicted Label        : {label} ({CLASS_NAMES[label]})\n")


if __name__ == "__main__":
    main()