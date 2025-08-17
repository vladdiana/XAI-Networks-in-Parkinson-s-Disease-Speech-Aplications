import os, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt, pandas as pd
BASE = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
TEST = os.path.join(BASE, "test_features_normalized.csv")
MODEL= os.path.join(BASE, "models", "speechnn_ddk.pth")
OUT  = os.path.join(BASE, "explainability", "gradients"); os.makedirs(OUT, exist_ok=True)

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

m = SpeechNN(len(mfcc)); m.load_state_dict(torch.load(MODEL, map_location="cpu")); m.eval()

def grad_imp(model, sample):
    s = sample.clone().unsqueeze(0); s.requires_grad_(True)
    out = model(s); tgt = out.argmax(1); score = out[0, tgt]; score.backward()
    return np.abs(s.grad.detach().cpu().numpy()[0])

imp = grad_imp(m, X[0])
plt.figure(figsize=(6,4)); plt.barh(mfcc, imp); plt.title("Gradient importance (sample 0)")
plt.tight_layout(); p1 = os.path.join(OUT,"gradients_sample0.png"); plt.savefig(p1); plt.close()

all_imps = [grad_imp(m, X[i]) for i in range(len(X))]
mean_imp = np.mean(np.stack(all_imps, axis=0), axis=0)
plt.figure(figsize=(6,4)); plt.barh(mfcc, mean_imp); plt.title("Global gradient importance (test mean)")
plt.tight_layout(); p2 = os.path.join(OUT,"gradients_global.png"); plt.savefig(p2); plt.close()

print(f"✅ Saved gradient importance:\n - {p1}\n - {p2}")
