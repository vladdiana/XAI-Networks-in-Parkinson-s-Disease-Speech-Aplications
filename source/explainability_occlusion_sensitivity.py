import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_nn import SpeechNN  # importăm clasa modelului tău

# 1. Load test data
test_df = pd.read_csv("test_features_normalized.csv")
X_test = test_df.drop(columns=["file_path", "label"]).values
y_test = test_df["label"].map({"HC": 0, "PD": 1}).values

# Convert to torch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 2. Load trained model
model = SpeechNN(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("nn_model.pth"))
model.eval()

# 3. Choose one sample
sample_idx = 0
original_sample = X_test_tensor[sample_idx].clone()
original_pred = torch.softmax(model(original_sample.unsqueeze(0)), dim=1)[0, 1].item()

# 4. Occlusion Sensitivity
n_features = X_test.shape[1]
occlusion_importance = []

for i in range(n_features):
    modified_sample = original_sample.clone()
    modified_sample[i] = 0.0  # mask one feature
    new_pred = torch.softmax(model(modified_sample.unsqueeze(0)), dim=1)[0, 1].item()
    impact = original_pred - new_pred
    occlusion_importance.append(impact)

# 5. Plot results
plt.figure(figsize=(10, 5))
plt.bar(range(n_features), occlusion_importance)
plt.xticks(range(n_features), [f"MFCC_{i+1}" for i in range(n_features)], rotation=45)
plt.ylabel("Prediction Impact (PD class)")
plt.title("Occlusion Sensitivity for MFCC features")
plt.tight_layout()
plt.savefig("source/reports/occlusion_sensitivity.png")
plt.close()
print("Occlusion sensitivity plot saved in 'source/reports/occlusion_sensitivity.png'")
