import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# ===== Paths =====
BASE_DIR = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\vowels"
TEST_CSV = os.path.join(BASE_DIR, "test_features_normalized.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "speechnn_vowels.pth")
OUT_DIR = os.path.join(BASE_DIR, "explainability", "gradients")
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
X_t = torch.tensor(X)

# Load model
model = SpeechNN(input_dim=len(mfcc_cols))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def gradient_importance(model, sample):
    sample = sample.clone().unsqueeze(0)
    sample.requires_grad_(True)
    out = model(sample)
    target = out.argmax(dim=1)
    score = out[0, target]
    score.backward()
    grads = sample.grad.detach().cpu().numpy()[0]
    return np.abs(grads)

# Single sample
imp = gradient_importance(model, X_t[0])

plt.figure(figsize=(6,4))
plt.barh(mfcc_cols, imp)
plt.title("Gradient-based importance (sample 0)")
plt.tight_layout()
png1 = os.path.join(OUT_DIR, "gradients_sample0.png")
plt.savefig(png1); plt.close()

# Global (mean over all test samples)
all_imps = []
for i in range(len(X_t)):
    all_imps.append(gradient_importance(model, X_t[i]))
mean_imp = np.mean(np.stack(all_imps, axis=0), axis=0)

plt.figure(figsize=(6,4))
plt.barh(mfcc_cols, mean_imp)
plt.title("Global Gradient-based importance (test mean)")
plt.tight_layout()
png2 = os.path.join(OUT_DIR, "gradients_global.png")
plt.savefig(png2); plt.close()

print(f"✅ Saved gradient importance to:\n - {png1}\n - {png2}")
