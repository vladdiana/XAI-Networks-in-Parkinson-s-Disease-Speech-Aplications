import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# === Configurations ===
input_dir = "gradcam_cnn_outputs"
output_path = "gradcam_cnn_outputs/COMPARATION_gradcam.png"
samples_per_class = 3

# === Separate images by class based on filename ===
hc_images = [f for f in os.listdir(input_dir) if f.lower().startswith("gradcam_hc")]
pd_images = [f for f in os.listdir(input_dir) if f.lower().startswith("gradcam_pd")]

# === Select random samples ===
hc_samples = random.sample(hc_images, min(samples_per_class, len(hc_images)))
pd_samples = random.sample(pd_images, min(samples_per_class, len(pd_images)))

# === Load and display ===
fig, axs = plt.subplots(2, samples_per_class, figsize=(samples_per_class * 3, 6))
fig.suptitle("Grad-CAM Comparison: HC (top) vs PD (bottom)", fontsize=14)

for i, fname in enumerate(hc_samples):
    img = Image.open(os.path.join(input_dir, fname))
    axs[0, i].imshow(img)
    axs[0, i].axis("off")
    axs[0, i].set_title(f"HC {i+1}")

for i, fname in enumerate(pd_samples):
    img = Image.open(os.path.join(input_dir, fname))
    axs[1, i].imshow(img)
    axs[1, i].axis("off")
    axs[1, i].set_title(f"PD {i+1}")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_path)
plt.close()
print(f"✅ Saved comparison figure to: {output_path}")
