import os, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt, pandas as pd
BASE = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
TEST = os.path.join(BASE, "test_features_normalized.csv")
MODEL= os.path.join(BASE, "models", "speechnn_ddk.pth")
OUT  = os.path.join(BASE, "explainability", "activations"); os.makedirs(OUT, exist_ok=True)

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
x = torch.tensor(df[mfcc].values[:1].astype(np.float32))
m = SpeechNN(len(mfcc)); m.load_state_dict(torch.load(MODEL, map_location="cpu")); m.eval()

acts={}
def hook(n):
    def _h(mod, i, o): acts[n]=o.detach().cpu().numpy()
    return _h
for i,layer in enumerate(m.net):
    if isinstance(layer, nn.Linear): layer.register_forward_hook(hook(f"Linear_{i}"))

_ = m(x)
for name, a in acts.items():
    plt.figure(figsize=(8,3)); plt.bar(range(a.shape[1]), a[0]); plt.title(f"Activation Map - {name}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{name}_activation.png")); plt.close()
print(f"✅ Saved activations to: {OUT}")
