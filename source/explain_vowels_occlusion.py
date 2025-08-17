import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# ===== Paths =====
BASE_DIR = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\vowels"
TEST_CSV = os.path.join(BASE_DIR, "test_features_normalized.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "speechnn_vowels.pth")
OUT_DIR = os.path.join(BASE_DIR, "explainability", "occlusion")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Model =====
class SpeechNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

# Load data
df = pd.read_csv(TEST_CSV)
mfcc_cols = [c for c in df.columns if c.startswith("mfcc_")]
X = df[mfcc_cols].values.astype(np.float32)
y = df["label"].map({"HC":0,"PD":1}).values
X_t = torch.tensor(X)
y_t = torch.tensor(y)

# Load model
model = SpeechNN(input_dim=len(mfcc_cols))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Pick one sample
idx = 0
x0 = X_t[idx].clone()
with torch.no_grad():
    base = F.softmax(model(x0.unsqueeze(0)), dim=1)[0, y_t[idx]].item()

impacts = []
for k in range(len(mfcc_cols)):
    x_mod = x0.clone()
    x_mod[k] = 0.0
    with torch.no_grad():
        p = F.softmax(model(x_mod.unsqueeze(0)), dim=1)[0, y_t[idx]].item()
    impacts.append(base - p)

plt.figure(figsize=(8,4))
plt.bar(range(len(mfcc_cols)), impacts)
plt.xticks(range(len(mfcc_cols)), mfcc_cols, rotation=45, ha='right')
plt.ylabel("Prediction impact (true class prob drop)")
plt.title("Occlusion Sensitivity (sample 0)")
plt.tight_layout()
out_png = os.path.join(OUT_DIR, "occlusion_sample0.png")
plt.savefig(out_png); plt.close()

print(f"✅ Saved occlusion sensitivity to: {out_png}")
