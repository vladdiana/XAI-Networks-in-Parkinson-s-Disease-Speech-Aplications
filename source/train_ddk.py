import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

BASE = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
TRAIN = os.path.join(BASE, "train_features_normalized.csv")
TEST  = os.path.join(BASE, "test_features_normalized.csv")
REPORTS = os.path.join(BASE, "reports"); os.makedirs(REPORTS, exist_ok=True)
MODELS  = os.path.join(BASE, "models");  os.makedirs(MODELS, exist_ok=True)

train_df = pd.read_csv(TRAIN); test_df = pd.read_csv(TEST)
mfcc = [c for c in train_df.columns if c.startswith("mfcc_")]
Xtr = train_df[mfcc].values.astype(np.float32); Xte = test_df[mfcc].values.astype(np.float32)
ytr = train_df["label"].map({"HC":0,"PD":1}).values.astype(np.int64)
yte = test_df["label"].map({"HC":0,"PD":1}).values.astype(np.int64)

Xtr_t, Xte_t = torch.tensor(Xtr), torch.tensor(Xte)
ytr_t, yte_t = torch.tensor(ytr), torch.tensor(yte)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xte_t, yte_t), batch_size=64, shuffle=False)

class SpeechNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

model = SpeechNN(Xtr.shape[1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(), lr=1e-3)
epochs, losses, accs = 30, [], []
device = next(model.parameters()).device

for ep in range(1, epochs+1):
    model.train(); run_loss=0; correct=0; total=0
    for xb,yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); out = model(xb); loss = crit(out,yb)
        loss.backward(); opt.step()
        run_loss += loss.item()
        correct += (out.argmax(1)==yb).sum().item(); total += yb.size(0)
    losses.append(run_loss/len(train_loader)); accs.append(correct/total)
    if ep in {1,5,10,15,20,25,30}: print(f"Epoch {ep:02d}/{epochs} | Loss {losses[-1]:.4f} | Acc {accs[-1]:.4f}")

# Eval
model.eval(); preds=[]
with torch.no_grad():
    for xb,_ in test_loader:
        xb = xb.to(device); preds.extend(model(xb).argmax(1).cpu().numpy())
acc = accuracy_score(yte, preds); print(f"\n✅ Test Accuracy (DDK): {acc:.4f}")

# Save outputs
rep_path = os.path.join(REPORTS, "SpeechNN_ddk_report.txt")
with open(rep_path, "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(yte, preds, target_names=["HC","PD"]))

cm = confusion_matrix(yte, preds)
plt.figure(figsize=(4.5,4)); plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix - SpeechNN (DDK)")
plt.xticks([0,1],["HC","PD"]); plt.yticks([0,1],["HC","PD"])
for i in range(2):
    for j in range(2): plt.text(j,i,cm[i,j],ha="center",va="center")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
cm_png = os.path.join(REPORTS,"SpeechNN_ddk_confusion_matrix.png"); plt.savefig(cm_png); plt.close()

# Curves
plt.figure(); plt.plot(range(1,epochs+1), accs, marker="o"); plt.grid(True)
plt.title("Training Accuracy - SpeechNN (DDK)"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
acc_png = os.path.join(REPORTS,"SpeechNN_ddk_train_accuracy.png"); plt.savefig(acc_png); plt.close()

plt.figure(); plt.plot(range(1,epochs+1), losses, marker="o"); plt.grid(True)
plt.title("Training Loss - SpeechNN (DDK)"); plt.xlabel("Epoch"); plt.ylabel("Loss")
loss_png = os.path.join(REPORTS,"SpeechNN_ddk_train_loss.png"); plt.savefig(loss_png); plt.close()

model_path = os.path.join(MODELS,"speechnn_ddk.pth"); torch.save(model.state_dict(), model_path)
print("\n💾 Saved:")
print(f" - Model: {model_path}")
print(f" - Report: {rep_path}")
print(f" - Confusion matrix: {cm_png}")
print(f" - Acc/Loss plots: {acc_png} , {loss_png}")
