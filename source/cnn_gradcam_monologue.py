# source/cnn_gradcam_monologue.py
import os
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt

# -------- Paths (adjust only if your project path changed) --------
DATA_VAL  = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\spectrograms\validation"
# Use the fine-tuned *best* checkpoint if you have it; otherwise, use the ft one
MODEL_PATH = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\models\cnn_monologue_resnet18_best.pth"
OUT_DIR   = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\explainability\gradcam"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- Preprocessing (match training) --------
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# -------- Build the model exactly like in training --------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# We need gradients on layer4 for Grad-CAM. Keep others frozen if you want.
for p in model.parameters():
    p.requires_grad = False
for p in model.layer4.parameters():
    p.requires_grad = True  # <-- essential for Grad-CAM

model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(128, 2)
)

# Load trained weights
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)

# Enable grad globally (just in case)
torch.set_grad_enabled(True)
model.eval()

# Attach Grad-CAM on the final conv block
cam_extractor = GradCAM(model, target_layer="layer4")

# -------- Run Grad-CAM on validation images --------
count = 0
for label in ["HC", "PD"]:
    folder = os.path.join(DATA_VAL, label)
    if not os.path.isdir(folder):
        print(f"⚠️ Missing folder: {folder}")
        continue

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue

        img_path = os.path.join(folder, fname)
        pil_img = Image.open(img_path).convert("RGB")

        # Preprocess
        input_tensor = tfm(pil_img).unsqueeze(0)

        # Forward pass (no torch.no_grad() here — we need grads)
        output = model(input_tensor)
        predicted_class = output.argmax().item()

        # Get CAM for the predicted class
        # torchcam returns a list of activation maps; take the first
        activation_map = cam_extractor(predicted_class, output)[0].detach().cpu()

        # Overlay heatmap onto the original image (resized to 224x224)
        base_img_224 = pil_img.resize((224, 224))
        cam_pil = to_pil_image(activation_map)
        result = overlay_mask(base_img_224, cam_pil, alpha=0.5)

        # Save
        out_name = f"gradcam_{label.lower()}_{fname}"
        out_path = os.path.join(OUT_DIR, out_name)
        plt.imshow(result)
        plt.axis("off")
        plt.title(f"{label} | Pred: {'HC' if predicted_class==0 else 'PD'}")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        count += 1

print(f"✅ Grad-CAM completed. Saved {count} overlays to:\n{OUT_DIR}")
