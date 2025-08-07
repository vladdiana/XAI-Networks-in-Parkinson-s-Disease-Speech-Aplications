import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ======= 1. Paths =======
train_dir = "source/reports/spectrograms/training"
val_dir = "source/reports/spectrograms/validation"
save_path = "source/reports/models/cnn_resnet18_finetuned.pth"

# ======= 2. Transforms =======
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # pretrained weights need 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# ======= 3. Datasets and Loaders =======
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# ======= 4. Model =======
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False  # freeze all layers

# Replace classifier head
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ======= 5. Loss and Optimizer =======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-5)

# ======= 6. Training loop =======
epochs = 15
val_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_accuracies.append(val_acc)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Val Acc: {val_acc:.2f}")

# ======= 7. Save model =======
torch.save(model.state_dict(), save_path)
print(f"✅ Fine-tuned model saved at: {save_path}")

# ======= 8. Save validation accuracy plot =======
accuracy_plot_path = os.path.join("accuracy_results/accuracy_finetuned_cnn.jpg")
os.makedirs(os.path.dirname(accuracy_plot_path), exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), val_accuracies, marker='o')
plt.title("Validation Accuracy during Fine-Tuning")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(accuracy_plot_path)
plt.close()

print(f"📈 Validation accuracy plot saved to: {accuracy_plot_path}")
