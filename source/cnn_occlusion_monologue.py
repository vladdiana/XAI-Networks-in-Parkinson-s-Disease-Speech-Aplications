import os, torch, numpy as np, matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn.functional as F

VAL_PD    = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\spectrograms\validation\PD"
MODEL_PATH= r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\models\cnn_monologue_resnet18.pth"
OUT_PNG   = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\explainability\occlusion\occlusion_sample.png"
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for p in m.parameters(): p.requires_grad=False
m.fc = torch.nn.Sequential(torch.nn.Linear(m.fc.in_features,128), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(128,2))
m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu")); m.eval()

# Take one PD image (you can switch PD -> HC above to test other class)
fname = next(f for f in os.listdir(VAL_PD) if f.lower().endswith(".png"))
img = Image.open(os.path.join(VAL_PD, fname)).convert("RGB")
ten = tfm(img).unsqueeze(0)

with torch.no_grad():
    base_prob = F.softmax(m(ten), dim=1)[0, m(ten).argmax(1)].item()

H,W = 224,224
heat = np.zeros((H,W), dtype=float)
step, size = 16, 32  # grid step and occlusion window

for y in range(0, H, step):
    for x in range(0, W, step):
        t = ten.clone()
        t[:, :, y:y+size, x:x+size] = 0
        with torch.no_grad():
            p = F.softmax(m(t), dim=1)[0, m(ten).argmax(1)].item()
        heat[y:y+size, x:x+size] = max(0, base_prob - p)

plt.figure(figsize=(4,4))
plt.imshow(img.resize((224,224))); plt.imshow(heat, cmap='hot', alpha=0.5)
plt.axis("off"); plt.title("Occlusion Sensitivity (Monologue)")
plt.tight_layout(); plt.savefig(OUT_PNG, bbox_inches='tight', pad_inches=0); plt.close()
print("✅ Saved:", OUT_PNG)
