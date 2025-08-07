import os
import torch
from torchvision import models, transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt

# === Set correct paths based on your project structure ===
base_dir = "source/reports/spectrograms/validation"
output_dir = "gradcam_cnn_outputs"


model_path = "source/reports/models/cnn_resnet18.pth"  # adjust path if needed

# === Create output directory if it doesn't exist ===
os.makedirs(output_dir, exist_ok=True)

# === Define image preprocessing ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert to 1 channel
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

# === Attach Grad-CAM to final convolutional layer ===
cam_extractor = GradCAM(model, target_layer="layer4")

# === Process images in HC and PD folders ===
image_count = 0
for label in ["HC", "PD"]:
    folder = os.path.join(base_dir, label)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue  # skip non-images

        img_path = os.path.join(folder, fname)
        pil_img = Image.open(img_path).convert("RGB")

        # Preprocess image
        input_tensor = transform(pil_img).unsqueeze(0)

        # Forward pass
        input_tensor.requires_grad_()
        output = model(input_tensor)
        predicted_class = output.argmax().item()

        # Grad-CAM heatmap
        activation_map = cam_extractor(predicted_class, output)

        # Overlay CAM on original image
        resized_img = transforms.Resize((224, 224))(pil_img)
        from torchvision.transforms.functional import to_pil_image

        # Convert tensor mask to PIL image before overlay
        cam_pil = to_pil_image(activation_map[0].detach().cpu())
        result = overlay_mask(resized_img, cam_pil, alpha=0.5)

        # Save Grad-CAM image
        out_name = f"gradcam_{label.lower()}_{fname}"
        out_path = os.path.join(output_dir, out_name)
        plt.imshow(result)
        plt.axis("off")
        plt.title(f"{label} | Predicted: {'HC' if predicted_class==0 else 'PD'}")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        image_count += 1

print(f"\n✅ Grad-CAM completed for {image_count} images.")
print(f"Results saved to: {output_dir}")
