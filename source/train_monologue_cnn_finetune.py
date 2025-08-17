import os, numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

# ===== Paths =====
DATA_DIR  = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\spectrograms"
OUT_BASE  = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue"
OUT_MODELS = os.path.join(OUT_BASE, "models");  os.makedirs(OUT_MODELS,  exist_ok=True)
OUT_REPORTS= os.path.join(OUT_BASE, "reports"); os.makedirs(OUT_REPORTS, exist_ok=True)

# ===== Custom SpecAugment-lite (time/freq masking on tensors) =====
import random, torch.nn.functional as F
class TimeFreqMask:
    """Apply up to K random time/frequency rectangular masks on a spectrogram tensor [3, H, W]."""
    def __init__(self, time_mask=0.15, freq_mask=0.20, num_masks=2):
        self.time_mask = time_mask  # fraction of width
        self.freq_mask = freq_mask  # fraction of height
        self.num_masks = num_masks
    def __call__(self, x):  # x: tensor CxHxW
        C, H, W = x.shape
        for _ in range(self.num_masks):
            # time mask (along width)
            t = int(W * self.time_mask * random.random())
            if t > 0:
                x[:, :, random.randint(0, max(0, W - t)) :][:, :, :t] = 0
            # freq mask (along height)
            f = int(H * self.freq_mask * random.random())
            if f > 0:
                x[:, random.randint(0, max(0, H - f)) :, :][:, :f, :] = 0
        return x

# ===== Transforms =====
base_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

train_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    TimeFreqMask(time_mask=0.20, freq_mask=0.25, num_masks=2),  # augmentation
    base_norm,
])

val_tfms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    base_norm,
])

# ===== Datasets & Loaders =====
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"training"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR,"validation"), transform=val_tfms)

# optional: weighted sampler (balanced anyway, but helps with small data)
class_counts = np.bincount([y for _, y in train_ds.samples])
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = [class_weights[y] for _, y in train_ds.samples]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

# ===== Model: unfreeze layer4 for fine-tuning =====
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
# unfreeze last block
for p in model.layer4.parameters():
    p.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Two optim groups: small LR for backbone, larger for head
backbone_params = [p for p in model.layer4.parameters() if p.requires_grad]
head_params = [p for p in model.fc.parameters() if p.requires_grad]
optimizer = optim.Adam([
    {"params": backbone_params, "lr": 1e-5},
    {"params": head_params,     "lr": 1e-4},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss()

epochs = 30
best_acc = 0.0
val_accs = []

for ep in range(1, epochs+1):
    model.train()
    run_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    scheduler.step()

    # validation
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(y.cpu().numpy())
    acc = (np.array(preds) == np.array(gts)).mean()
    val_accs.append(acc)
    print(f"Epoch {ep:02d}/{epochs} | Loss {run_loss/len(train_loader):.4f} | ValAcc {acc:.4f}")

    # save best
    if acc >= best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(OUT_MODELS, "cnn_monologue_resnet18_best.pth"))

# final save
final_path = os.path.join(OUT_MODELS, "cnn_monologue_resnet18_ft.pth")
torch.save(model.state_dict(), final_path)
print("✅ Saved model:", final_path, "(best:", best_acc, ")")

# reports
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:\n", classification_report(gts, preds, target_names=val_ds.classes))
cm = confusion_matrix(gts, preds)
plt.figure(figsize=(5,4)); plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - CNN (Monologue, FT)")
plt.xticks([0,1], val_ds.classes); plt.yticks([0,1], val_ds.classes)
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha="center",va="center")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
cm_png = os.path.join(OUT_REPORTS, "cnn_monologue_confusion_matrix_ft.png")
plt.savefig(cm_png); plt.close()

plt.figure(); plt.plot(range(1,epochs+1), val_accs, marker='o'); plt.grid(True)
plt.title("Validation Accuracy - CNN (Monologue, FT)"); plt.xlabel("Epoch"); plt.ylabel("Acc")
acc_png = os.path.join(OUT_REPORTS, "cnn_monologue_val_acc_ft.png")
plt.savefig(acc_png); plt.close()
print("💾 Saved:", cm_png, "and", acc_png)
