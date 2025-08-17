import os, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DATA_DIR  = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\spectrograms"
OUT_BASE  = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue"
OUT_MODELS = os.path.join(OUT_BASE, "models");  os.makedirs(OUT_MODELS,  exist_ok=True)
OUT_REPORTS= os.path.join(OUT_BASE, "reports"); os.makedirs(OUT_REPORTS, exist_ok=True)

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"training"),   transform=tfm)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR,"validation"), transform=tfm)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Sequential(nn.Linear(model.fc.in_features,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
crit = nn.CrossEntropyLoss()
opt  = optim.Adam(model.fc.parameters(), lr=1e-4)
epochs = 15

val_accs=[]; preds=[]; gts=[]
for ep in range(1, epochs+1):
    model.train(); run_loss=0.0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); out = model(x); loss = crit(out,y)
        loss.backward(); opt.step()
        run_loss += loss.item()
    # validate
    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy()); gts.extend(y.cpu().numpy())
    acc = (np.array(preds)==np.array(gts)).mean(); val_accs.append(acc)
    print(f"Epoch {ep:02d}/{epochs} | Loss {run_loss/len(train_loader):.4f} | ValAcc {acc:.4f}")

model_path = os.path.join(OUT_MODELS, "cnn_monologue_resnet18.pth")
torch.save(model.state_dict(), model_path); print("✅ Saved model:", model_path)

print("Classification Report:\n", classification_report(gts, preds, target_names=train_ds.classes))
cm = confusion_matrix(gts, preds)
plt.figure(figsize=(5,4)); plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - CNN (Monologue)")
plt.xticks([0,1], train_ds.classes); plt.yticks([0,1], train_ds.classes)
for i in range(2):
    for j in range(2): plt.text(j,i,cm[i,j],ha='center',va='center')
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
cm_png = os.path.join(OUT_REPORTS, "cnn_monologue_confusion_matrix.png")
plt.savefig(cm_png); plt.close()

plt.figure(); plt.plot(range(1,epochs+1), val_accs, marker='o'); plt.grid(True)
plt.title("Validation Accuracy - CNN (Monologue)"); plt.xlabel("Epoch"); plt.ylabel("Acc")
acc_png = os.path.join(OUT_REPORTS, "cnn_monologue_val_acc.png")
plt.savefig(acc_png); plt.close()
print("💾 Saved:", cm_png, "and", acc_png)
