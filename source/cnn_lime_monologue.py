import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision.transforms.functional import to_pil_image

# -------- Paths --------
VAL_DIR    = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\spectrograms\validation"
MODEL_PATH = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\models\cnn_monologue_resnet18_best.pth"
OUT_DIR    = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\monologue\explainability\lime"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- Preprocessing (must match training) --------
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# -------- Build the model exactly like in training --------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(128, 2)
)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

# Class order in ImageFolder is alphabetical: ["HC", "PD"] -> index 0 = HC, 1 = PD
class_names = ["HC", "PD"]

# -------- Prediction function for LIME (expects numpy images -> proba matrix) --------
def predict_proba(imgs: np.ndarray) -> np.ndarray:
    """
    imgs: array of shape (N, H, W, 3) in uint8 or float in [0,1]
    return: (N, 2) probabilities
    """
    batch = []
    for arr in imgs:
        # Convert to PIL, then apply same transforms as training
        pil = Image.fromarray((arr * 255).astype(np.uint8)) if arr.max() <= 1.0 else Image.fromarray(arr.astype(np.uint8))
        ten = tfm(pil).unsqueeze(0)
        batch.append(ten)
    x = torch.cat(batch, dim=0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs

# -------- Helper: explain 1 image and save result --------
def explain_image(img_path: str, out_prefix: str, num_samples: int = 1000):
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)  # HxWx3 in uint8

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=np_img,
        classifier_fn=predict_proba,
        top_labels=1,            # explain only the predicted class
        hide_color=0,
        num_samples=num_samples  # more samples -> smoother but slower
    )

    # Build a pretty visualization of the most important superpixels (positive only)
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label=top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Overlay LIME mask boundaries on the original image
    vis = mark_boundaries(temp / 255.0, mask)

    # Save figure
    plt.figure(figsize=(4,4))
    plt.imshow(vis)
    plt.axis("off")
    plt.title(f"LIME | Predicted: {class_names[top_label]}")
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_lime.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("✅ Saved:", out_path)

# -------- Run LIME on a few HC and PD images --------
def pick_first_image(folder):
    for f in os.listdir(folder):
        if f.lower().endswith(".png"):
            return os.path.join(folder, f)
    return None

samples_per_class = 2  # change if you want more
for label in ["HC", "PD"]:
    folder = os.path.join(VAL_DIR, label)
    if not os.path.isdir(folder):
        print("⚠️ Missing folder:", folder)
        continue

    # Take up to 'samples_per_class' images
    count = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue
        img_path = os.path.join(folder, fname)
        out_prefix = f"monologue_{label.lower()}_{os.path.splitext(fname)[0]}"
        explain_image(img_path, out_prefix, num_samples=1000)
        count += 1
        if count >= samples_per_class:
            break

print(f"Done. LIME explanations saved in: {OUT_DIR}")
