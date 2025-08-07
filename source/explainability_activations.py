import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train_nn import SpeechNN  # asigură-te că modelul SpeechNN este definit în train_nn.py
import pandas as pd

# ============ 1. Create output folder ============
os.makedirs("source/reports/activations", exist_ok=True)

# ============ 2. Load trained model ============
model = SpeechNN(input_dim=13)  # avem 13 caracteristici MFCC
model.load_state_dict(torch.load("nn_model.pth", map_location=torch.device("cpu")))
model.eval()

# ============ 3. Load a test sample ============
test_data = pd.read_csv("test_features_normalized.csv")
X_test = test_data.drop(columns=["file_path", "label"]).values
sample = torch.tensor(X_test[0].reshape(1, -1), dtype=torch.float32)

# ============ 4. Hook function to store activations ============
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

# Attach hooks to all Linear layers
for idx, layer in enumerate(model.layers):
    if isinstance(layer, torch.nn.Linear):
        layer.register_forward_hook(get_activation(f"Layer_{idx}"))

# ============ 5. Forward pass to collect activations ============
_ = model(sample)

# ============ 6. Plot and save activation maps ============
for name, act in activations.items():
    plt.figure(figsize=(6, 3))
    plt.bar(range(act.shape[1]), act[0], color="skyblue")
    plt.title(f"Activation Map - {name}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Value")
    plt.tight_layout()
    plt.savefig(f"source/reports/activations/{name}_activation.png")
    plt.close()

print("Activation maps saved in 'source/reports/activations/' folder.")
