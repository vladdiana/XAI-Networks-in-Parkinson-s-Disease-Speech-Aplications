import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ============= PATHS (ABSOLUTE) =============
# If your project folder is different, change ONLY the line below:
BASE_DIR = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\vowels"

TRAIN_CSV = os.path.join(BASE_DIR, "train_features_normalized.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test_features_normalized.csv")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============= 1) LOAD DATA =============
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Keep only MFCC columns as features; 'label' stays as target
mfcc_cols = [c for c in train_df.columns if c.startswith("mfcc_")]
X_train = train_df[mfcc_cols].values.astype(np.float32)
X_test  = test_df[mfcc_cols].values.astype(np.float32)

# Map labels to integers: HC -> 0, PD -> 1
y_train = train_df["label"].map({"HC": 0, "PD": 1}).values.astype(np.int64)
y_test  = test_df["label"].map({"HC": 0, "PD": 1}).values.astype(np.int64)

# Convert to torch tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

# Simple TensorDataset & DataLoader
train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
test_ds  = torch.utils.data.TensorDataset(X_test_t, y_test_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

# ============= 2) DEFINE MODEL =============
class SpeechNN(nn.Module):
    """Small fully-connected network for MFCC vectors."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)  # 2 classes: HC, PD
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]
model = SpeechNN(input_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ============= 3) TRAIN =============
epochs = 30
train_losses = []
train_accuracies = []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    avg_loss = running_loss / len(train_loader)
    acc = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(acc)

    if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
        print(f"Epoch {epoch:02d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Train Acc: {acc:.4f}")

# ============= 4) EVALUATE =============
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

acc_test = accuracy_score(y_test, all_preds)
print(f"\n✅ Test Accuracy (VOWELS): {acc_test:.4f}")

# Save classification report
report_txt = os.path.join(REPORTS_DIR, "SpeechNN_vowels_report.txt")
with open(report_txt, "w") as f:
    f.write(f"Test Accuracy: {acc_test:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, all_preds, target_names=["HC", "PD"]))

# Save confusion matrix
cm = confusion_matrix(y_test, all_preds)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - SpeechNN (Vowels)")
plt.xticks([0, 1], ["HC", "PD"])
plt.yticks([0, 1], ["HC", "PD"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, "SpeechNN_vowels_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# Save training curves
plt.figure()
plt.plot(range(1, epochs + 1), train_accuracies, marker="o")
plt.title("Training Accuracy - SpeechNN (Vowels)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
acc_plot = os.path.join(REPORTS_DIR, "SpeechNN_vowels_train_accuracy.png")
plt.savefig(acc_plot)
plt.close()

plt.figure()
plt.plot(range(1, epochs + 1), train_losses, marker="o")
plt.title("Training Loss - SpeechNN (Vowels)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
loss_plot = os.path.join(REPORTS_DIR, "SpeechNN_vowels_train_loss.png")
plt.savefig(lossp:=os.path.join(REPORTS_DIR, "SpeechNN_vowels_train_loss.png"))
plt.close()

# Save model
model_path = os.path.join(MODELS_DIR, "speechnn_vowels.pth")
torch.save(model.state_dict(), model_path)

print("\n💾 Saved:")
print(f" - Model: {model_path}")
print(f" - Report: {report_txt}")
print(f" - Confusion matrix: {cm_path}")
print(f" - Acc/Loss plots: {acc_plot} , {lossp}")
