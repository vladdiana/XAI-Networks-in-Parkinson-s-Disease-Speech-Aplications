import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt, pandas as pd
BASE = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
TEST = os.path.join(BASE, "test_features_normalized.csv")
MODEL= os.path.join(BASE, "models", "speechnn_ddk.pth")
OUT  = os.path.join(BASE, "explainability", "occlusion"); os.makedirs(OUT, exist_ok=True)

class SpeechNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32,2)
        )
    def forward(self,x): return self.net(x)

df = pd.read_csv(TEST); mfcc = [c for c in df.columns if c.startswith("mfcc_")]
X = torch.tensor(df[mfcc].values.astype(np.float32))
y = torch.tensor(df["label"].map({"HC":0,"PD":1}).values)

m = SpeechNN(len(mfcc)); m.load_state_dict(torch.load(MODEL, map_location="cpu")); m.eval()

idx = 0; x0 = X[idx].clone()
with torch.no_grad():
    base = F.softmax(m(x0.unsqueeze(0)), dim=1)[0, y[idx]].item()

impacts=[]
for k in range(len(mfcc)):
    xm = x0.clone(); xm[k]=0.0
    with torch.no_grad():
        p = F.softmax(m(xm.unsqueeze(0)), dim=1)[0, y[idx]].item()
    impacts.append(base - p)

plt.figure(figsize=(8,4)); plt.bar(range(len(mfcc)), impacts)
plt.xticks(range(len(mfcc)), mfcc, rotation=45, ha='right')
plt.ylabel("Prediction impact (true class prob drop)")
plt.title("Occlusion Sensitivity (sample 0)")
plt.tight_layout(); outp = os.path.join(OUT,"occlusion_sample0.png"); plt.savefig(outp); plt.close()
print(f"✅ Saved occlusion sensitivity to: {outp}")
