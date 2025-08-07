import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ======== Configuration ========
data_dir = "source/reports/spectrograms"
batch_size = 16
epochs = 15
learning_rate = 0.0001
num_classes = 2  # HC and PD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Transforms for input images ========
from torchvision.transforms import Grayscale

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # <-- adaugă asta
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ======== Load datasets ========
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "training"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "validation"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ======== Load pre-trained ResNet18 and adapt it ========
model = models.resnet18(pretrained=True)

# If images are grayscale, change first conv layer to accept 1 channel
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Replace final FC layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# ======== Loss & optimizer ========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ======== Training loop ========
train_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_accuracies.append(acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_losses[-1]:.4f} | Val Acc: {acc:.4f}")

# ======== Save model ========
os.makedirs("source/reports/models", exist_ok=True)
torch.save(model.state_dict(), "source/reports/models/cnn_resnet18.pth")
print("\n✅ Model saved to source/reports/models/cnn_resnet18.pth")

# ======== Evaluation ========
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks([0, 1], train_dataset.classes)
plt.yticks([0, 1], train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig("source/reports/confusion_matrix_cnn.png")
plt.close()
print("Confusion matrix saved to source/reports/confusion_matrix_cnn.png")

# ======== Plot accuracy vs epochs ========
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), val_accuracies, marker='o', label='Validation Accuracy')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.savefig("accuracy_results/accuracy_train_cnn.png")
plt.close()
print("📈 Accuracy plot saved to source/reports/accuracy_plot.png")
