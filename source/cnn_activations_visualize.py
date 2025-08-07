import os, torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Config
model_path = "source/reports/models/cnn_resnet18.pth"
image_path = "source/reports/spectrograms/validation/PD"
output_dir = "source/reports/activations_cnn"
os.makedirs(output_dir, exist_ok=True)

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

# Hook-uri pentru layere convoluționale
activations = []
def hook_fn(module, inp, out): activations.append(out.detach().cpu())

model.layer1.register_forward_hook(hook_fn)
model.layer2.register_forward_hook(hook_fn)
img = Image.open(os.path.join(image_path, next(f for f in os.listdir(image_path) if f.endswith(".png")))).convert("RGB")
tensor = transform(img).unsqueeze(0)

_ = model(tensor)

# Vizualizează primele caracteristici din fiecare layer
for idx, act in enumerate(activations):
    fmap = act[0]
    fig, axes = plt.subplots(1, 5, figsize=(12,3))
    for i in range(5):
        axes[i].imshow(fmap[i], cmap='viridis')
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/activations_layer{idx+1}.png")
    plt.close()

print("CNN activations saved to:", output_dir)
