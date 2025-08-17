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
OUT_DIR = os.path.join(BASE_DIR, "explainability", "activations")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Model (must match train_vowels.py: uses self.net) =====
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
sample = torch.tensor(X[0:1])

# Load model
model = SpeechNN(input_dim=len(mfcc_cols))
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)  # keys match: 'net.0.weight', etc.
model.eval()

# Collect activations via forward hooks (on model.net layers)
activations = {}
def hook(name):
    def _hook(m, i, o):
        activations[name] = o.detach().cpu().numpy()
    return _hook

for idx, layer in enumerate(model.net):
    if isinstance(layer, nn.Linear):
        layer.register_forward_hook(hook(f"Linear_{idx}"))

# Forward pass to populate activations
_ = model(sample)

# Plot activations per Linear layer
for name, act in activations.items():
    plt.figure(figsize=(8, 3))
    plt.bar(range(act.shape[1]), act[0])
    plt.title(f"Activation Map - {name}")
    plt.xlabel("Neuron index")
    plt.ylabel("Activation")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{name}_activation.png")
    plt.savefig(out_png)
    plt.close()

print(f"✅ Saved activations to: {OUT_DIR}")
