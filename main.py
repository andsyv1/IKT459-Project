import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from Dataloader import HumidityDataset
from models import get_model

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
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dataset
dataset = HumidityDataset(DATA_DIR, CATEGORIES, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select model
model_name = "resnet"  # or "vgg", "densenet", "mobilenet"
model = get_model(model_name, len(CATEGORIES))
model = model.to(device)



# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Tracking lists
loss_history = []
accuracy_history = []
all_preds = []
all_labels = []

# Training Loop
for epoch in range(EPOCHS):
    running_loss = 0.0
    epoch_preds = []
    epoch_labels = []

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        epoch_preds.extend(preds.cpu().numpy())
        epoch_labels.extend(labels.cpu().numpy())


    # Track loss & accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = metrics.accuracy_score(epoch_labels, epoch_preds)
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_acc)

    all_preds.extend(epoch_preds)
    all_labels.extend(epoch_labels)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("Training Completed!")

# Compute final metrics
conf_matrix = metrics.confusion_matrix(all_labels, all_preds)
accuracy = metrics.accuracy_score(all_labels, all_preds)
report = metrics.classification_report(all_labels, all_preds, target_names=CATEGORIES)

print(f"Final Accuracy: {accuracy:.4f}")
print(f"Classification Report ({model_name}):\n{report}")

# Plot and save confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Confusion_matrix.jpg")
plt.show()

# Plot and save loss/accuracy graph
plt.figure(figsize=(10, 6))
epochs_range = range(1, EPOCHS + 1)
plt.plot(epochs_range, loss_history, label="Loss")
plt.plot(epochs_range, accuracy_history, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_accuracy_plot.jpg")
plt.show()
