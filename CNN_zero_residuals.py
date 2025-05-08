import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from Dataloader import HumidityDataset


# 🔥 Simplified CNN Model Without Residual Blocks
class HumidityCNN(nn.Module):
    def __init__(self, num_classes):
        super(HumidityCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Set parameters
DATA_DIR = "./Dataset"
CATEGORIES = [
    "1.1 Water bottel whit cold water",
    "1.2 Water bottel whit rests of cold water",
    "1.3 Water bottel whit no water",
    "1.4 Water bottel whit body temp water",
    "1.5 Water bottel whit rest of body temp water",
    "2.6 Cottage Cheese before eating",
    "2.7 Cottage Cheese after eating",
    "3.8 Shower apartment fungus",
    "3.9 Shower apartment less fungus",
    "4.10 Basement humidity",
    "4.11 Basement humidity",
    "5.12 window 1",
    "5.13 window 2",
    "5.14 window 3",
    "5.15 window 4",
    "6.16 punctured window 1",
    "6.17 punctured window 2",
    "6.18 punctured window 3",
    "6.19 punctured window 4",
    "7.20 Bedrom window cold spot",
    "8.21 Cold spot wall",
    "9.22 Cold spots, old window"
]

BATCH_SIZE = 184
EPOCHS = 100
LEARNING_RATE = 0.0001

# Data Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Dataset
dataset = HumidityDataset(DATA_DIR, CATEGORIES, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HumidityCNN(num_classes=len(CATEGORIES)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# History trackers
loss_history = []
accuracy_history = []
all_preds = []
all_labels = []

# Training Loop
for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()

    epoch_preds = []
    epoch_labels = []

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        epoch_preds.extend(preds.cpu().numpy())
        epoch_labels.extend(labels.cpu().numpy())

    scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = metrics.accuracy_score(epoch_labels, epoch_preds)
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)

    all_preds.extend(epoch_preds)
    all_labels.extend(epoch_labels)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

print("Training Completed!")

# Compute metrics
conf_matrix = metrics.confusion_matrix(all_labels, all_preds)
accuracy = metrics.accuracy_score(all_labels, all_preds)
report = metrics.classification_report(all_labels, all_preds, target_names=CATEGORIES)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("Confusion_matrix.jpg")
plt.show()

# Plot loss and accuracy
plt.figure(figsize=(10, 6))
epochs_range = range(1, EPOCHS + 1)
plt.plot(epochs_range, loss_history, label='Loss')
plt.plot(epochs_range, accuracy_history, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_accuracy_plot.jpg")
plt.show()

# Final reports
print(f"Final Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

