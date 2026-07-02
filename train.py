# 1. Imports & config
import os, json, pickle, time, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


DATA_PATH = "dataset.csv"
DRIVE_DIR = "model"  
os.makedirs(DRIVE_DIR, exist_ok=True)

BATCH_SIZE, EPOCHS, PATIENCE, LR, SEED = 32, 20, 5, 1e-3, 42

EMBED_DIM, CNN_FILTERS, LSTM_HIDDEN, DENSE_DIM, DROPOUT = 64, 64, 128, 128, 0.5

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")



# 2. Character tokenizer
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



# 3. Model: Embedding -> Multi-Scale CNN -> BiLSTM -> Manual Attention -> MLP
class Attention(nn.Module):
    """Hand-written additive attention (no library attention module)."""
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



# 4. Load data, build vocab (train split only), make loaders
df = pd.read_csv(DATA_PATH).dropna(subset=["payload", "label"])
df["payload"], df["label"] = df["payload"].astype(str), df["label"].astype(int)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
print(f"[INFO] Train: {len(train_df)}  Val: {len(val_df)}")

max_len = int(np.clip(np.percentile(train_df["payload"].str.len(), 95), 16, 512))
tok = CharTokenizer(max_len)
tok.fit(train_df["payload"].tolist())
print(f"[INFO] max_len={max_len}  vocab_size={tok.vocab_size}")

def make_loader(d, shuffle):
    X = torch.tensor(tok.encode_batch(d["payload"].tolist()), dtype=torch.long)
    y = torch.tensor(d["label"].tolist(), dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader, val_loader = make_loader(train_df, True), make_loader(val_df, False)



# 5. Train (Adam + BCELoss + early stopping, best-model-only checkpoint)
model = PayloadClassifier(tok.vocab_size, EMBED_DIM, CNN_FILTERS, LSTM_HIDDEN, DENSE_DIM, DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.BCELoss()

def run(loader, train):
    model.train() if train else model.eval()
    total, probs, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for X, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            X, y = X.to(device), y.to(device)
            if train: opt.zero_grad()
            p = model(X); loss = crit(p, y)
            if train: loss.backward(); opt.step()
            total += loss.item() * len(y)
            probs += p.detach().cpu().tolist()
            labels += y.detach().cpu().tolist()
    return total / len(loader.dataset), np.array(probs), np.array(labels)

best_loss, patience_ctr, best_state = float("inf"), 0, None
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("[INFO] Training...")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tl, tp, ty = run(train_loader, True)
    vl, vp, vy = run(val_loader, False)
    ta, va = accuracy_score(ty, tp >= 0.5), accuracy_score(vy, vp >= 0.5)
    history["train_loss"].append(tl); history["val_loss"].append(vl)
    history["train_acc"].append(ta); history["val_acc"].append(va)

    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tl:.4f} val_loss={vl:.4f} "
          f"train_acc={ta:.4f} val_acc={va:.4f} ({time.time()-t0:.1f}s)")

    if vl < best_loss - 1e-4:
        best_loss, patience_ctr = vl, 0
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, os.path.join(DRIVE_DIR, "best_model.pth"))
        print(f"   -> new best model saved (val_loss={vl:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
model.eval()



# 6. Final evaluation (best model)
_, vp, vy = run(val_loader, False)
vpred = (vp >= 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(vy, vpred),
    "precision": precision_score(vy, vpred, zero_division=0),
    "recall": recall_score(vy, vpred, zero_division=0),
    "f1": f1_score(vy, vpred, zero_division=0),
    "roc_auc": roc_auc_score(vy, vp),
}
cm = confusion_matrix(vy, vpred)

print("\n[FINAL METRICS]", {k: round(v, 4) for k, v in metrics.items()})
print("\n[CONFUSION MATRIX]\n", cm)
print("\n[CLASSIFICATION REPORT]\n", classification_report(vy, vpred, target_names=["Benign", "Malicious"]))



# 7. Save model + tokenizer + config
with open(os.path.join(DRIVE_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tok, f)

config = {"vocab_size": tok.vocab_size, "max_len": max_len, "embed_dim": EMBED_DIM,
          "cnn_filters": CNN_FILTERS, "lstm_hidden": LSTM_HIDDEN, "dense_dim": DENSE_DIM,
          "dropout": DROPOUT, "pad_idx": 0}
with open(os.path.join(DRIVE_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n[INFO] Saved best_model.pth, tokenizer.pkl, config.json to: {DRIVE_DIR}")


# 8. Save fugure: loss/accuracy curves, confusion matrix, ROC
def show_or_save(fig, filename):
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False
 
    if in_notebook:
        plt.show()
    else:
        path = os.path.join(DRIVE_DIR, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved figure: {path}")
 
fig1, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(history["train_acc"], label="Train"); axes[1].plot(history["val_acc"], label="Val")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout(); show_or_save(fig1, "loss_accuracy.png")
 
fig2 = plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix"); plt.colorbar()
plt.xticks([0, 1], ["Benign", "Malicious"]); plt.yticks([0, 1], ["Benign", "Malicious"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); show_or_save(fig2, "confusion_matrix.png")
 
fpr, tpr, _ = roc_curve(vy, vp)
fig3 = plt.figure(figsize=(4.5, 4.5))
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.tight_layout(); show_or_save(fig3, "roc_curve.png")