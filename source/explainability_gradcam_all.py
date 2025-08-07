import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from train_nn import SpeechNN  # Import the neural network class

# ==========================
# 1. Load the test dataset
# ==========================
test_df = pd.read_csv("test_features_normalized.csv")
X_test = test_df.drop(columns=["file_path", "label"]).values
feature_names = test_df.drop(columns=["file_path", "label"]).columns

# Convert to PyTorch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# ==========================
# 2. Load the trained model
# ==========================
model = SpeechNN(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("nn_model.pth", map_location=torch.device("cpu")))
model.eval()

# ==========================
# 3. Grad-CAM function
# ==========================
def gradcam_sample(model, sample):
    sample = sample.unsqueeze(0)  # Add batch dimension
    sample.requires_grad = True

    output = model(sample)
    pred_class = output.argmax(dim=1)
    pred_score = output[0, pred_class]
    pred_score.backward()

    grads = sample.grad.abs().squeeze().detach().numpy()
    return grads

# ==========================
# 4. Compute Grad-CAM for all test samples
# ==========================
all_importances = []
for i in range(len(X_test_tensor)):
    grads = gradcam_sample(model, X_test_tensor[i])
    all_importances.append(grads)

all_importances = np.array(all_importances)
mean_importances = all_importances.mean(axis=0)

# ==========================
# 5. Save and plot global importance
# ==========================
os.makedirs("source/reports", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.barh(feature_names, mean_importances)
plt.xlabel("Average Importance")
plt.title("Global Grad-CAM Importance (Test Set)")
plt.tight_layout()
plt.savefig("source/reports/gradcam_global_importance.png")
plt.close()

print("Grad-CAM global importance saved to 'source/reports/gradcam_global_importance.png'")
