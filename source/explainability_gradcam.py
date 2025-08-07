import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================
# 1. Define SpeechNN (same as in train_nn.py)
# =====================
class SpeechNN(nn.Module):
    def __init__(self, input_dim):
        super(SpeechNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

# =====================
# 2. Load Model
# =====================
INPUT_DIM = 13  # change if needed
model = SpeechNN(input_dim=INPUT_DIM)
model.load_state_dict(torch.load("nn_model.pth", map_location=torch.device("cpu")))
model.eval()

# =====================
# 3. Load test data
# =====================
test_data = pd.read_csv("test_features_normalized.csv")
X_test = test_data.drop(columns=["file_path", "label"]).values
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# =====================
# 4. Grad-CAM for tabular data
# =====================
def gradcam(model, input_tensor, target_class=None):
    input_tensor.requires_grad = True
    outputs = model(input_tensor)
    if target_class is None:
        target_class = outputs.argmax(dim=1)

    # Take the score for the target class
    score = outputs[0, target_class]
    score.backward()

    gradients = input_tensor.grad[0].numpy()
    importance = np.abs(gradients)

    return importance

# Example: explain first test sample
sample = X_test_tensor[0].unsqueeze(0)
importance = gradcam(model, sample)

# =====================
# 5. Plot feature importance
# =====================
import os
import os
feature_names = [f"mfcc_{i+1}" for i in range(importance.shape[0])]
plt.barh(feature_names, importance)
plt.xlabel("Importance")
plt.title("Grad-CAM (Gradient Importance) for Sample 0")
plt.tight_layout()
plt.savefig("source/reports/gradcam_sample0.png")
plt.close()
print("Grad-CAM sample 0 plot saved to 'source/reports/gradcam_sample0.png'")

