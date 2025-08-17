import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================
# 1. Load normalized data
# =====================
train_path = "train_features_normalized.csv"
test_path = "test_features_normalized.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["file_path", "label"]).values
y_train = train_df["label"].map({"HC": 0, "PD": 1}).values

X_test = test_df.drop(columns=["file_path", "label"]).values
y_test = test_df["label"].map({"HC": 0, "PD": 1}).values

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =====================
# 2. Define Neural Network
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

model = SpeechNN(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 3. Training Loop with Accuracy and Loss Tracking
# =====================
epochs = 30
train_accuracies = []
train_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_accuracies.append(train_acc)
    train_losses.append(avg_loss)

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

# =====================
# 4. Evaluation
# =====================
model.eval()
with torch.no_grad():
    y_pred = []
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =====================
# 5. Save the model
# =====================
torch.save(model.state_dict(), "nn_model.pth")
print("Model saved as nn_model.pth")

# =====================
# 6. Plot and Save Accuracy and Loss
# =====================
report_dir = os.path.join("accuracy_results")
os.makedirs(report_dir, exist_ok=True)

# Accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracies, marker='o')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
accuracy_path = os.path.join(report_dir, "accuracy_train_nn.jpg")
plt.savefig(accuracy_path)
plt.close()

# Loss plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, marker='o', color='red')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
loss_path = os.path.join(report_dir, "loss_train_nn.jpg")
plt.savefig(loss_path)
plt.close()

print(f"Accuracy plot saved to {accuracy_path}")
print(f"Loss plot saved to {loss_path}")
