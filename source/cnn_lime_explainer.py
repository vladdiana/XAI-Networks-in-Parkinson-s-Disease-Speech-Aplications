import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# === Paths ===
image_dir = "source/reports/spectrograms/validation/PD"
model_path = "source/reports/models/cnn_resnet18.pth"
output_dir = "source/reports/lime_cnn"
os.makedirs(output_dir, exist_ok=True)

# === Find first valid image ===
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
if not image_files:
    raise FileNotFoundError("No .png images found in PD folder.")

image_path = os.path.join(image_dir, image_files[0])
print(f"Using image: {image_path}")

# === Image preprocessing (must match training) ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Load trained CNN model ===
model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# === Define prediction function for LIME ===
def predict(images):
    model.eval()
    batch = []
    for img in images:
        img_pil = Image.fromarray(img).convert("L")
        input_tensor = transform(img_pil).unsqueeze(0)
        batch.append(input_tensor)
    batch_tensor = torch.cat(batch)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).numpy()
    return probs

# === Run LIME ===
img = np.array(Image.open(image_path).convert("RGB"))
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img, predict, top_labels=1, hide_color=0, num_samples=1000)

# === Visualize and save ===
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
)

lime_result = mark_boundaries(temp / 255.0, mask)

plt.imshow(lime_result)
plt.axis("off")
plt.title("LIME Explanation (CNN)")
output_file = os.path.join(output_dir, "lime_cnn_result.png")
plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"\n✅ LIME explanation saved to: {output_file}")
