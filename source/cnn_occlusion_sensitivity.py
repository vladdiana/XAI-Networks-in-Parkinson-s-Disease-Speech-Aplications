import os, torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

# Config
model_path = "source/reports/models/cnn_resnet18.pth"
image_folder = "source/reports/spectrograms/validation/PD"
output_path = "source/reports/occlusion_cnn.png"

# Preprocesare
transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224,224)),
    transforms.ToTensor(), transforms.Normalize([0.5],[0.5])
])

# Încarcă model
model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# adding an image
fname = next(f for f in os.listdir(image_folder) if f.endswith(".png"))
img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(tensor)
    pred = out.argmax(dim=1).item()

H, W = tensor.shape[2:]
heatmap = np.zeros((H, W))
step, size = 15, 30

for y in range(0, H, step):
    for x in range(0, W, step):
        t = tensor.clone()
        t[:, :, y:y+size, x:x+size] = 0
        with torch.no_grad():
            o = model(t)
            prob = F.softmax(o, dim=1)[0, pred].item()
        heatmap[y:y+size, x:x+size] = 1 - prob

plt.imshow(img.resize((224,224)))
plt.imshow(heatmap, cmap='hot', alpha=0.5)
plt.axis("off")
plt.title("Occlusion Sensitivity")
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close()
print("Occlusion sensitivity saved:", output_path)
