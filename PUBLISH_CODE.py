import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from Dataloader import HumidityDataset

# ---------- Residual Block ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = torch.add(x, residual)
        return nn.ReLU()(output)

# ---------- Main Model ----------
class HumidityCNN(nn.Module):
    def __init__(self, num_classes):
        super(HumidityCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64, 128, stride=2)
        self.res2 = ResidualBlock(128, 256, stride=2)
        self.res3 = ResidualBlock(256, 512, stride=2)
        self.res4 = ResidualBlock(512, 1024, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.variance_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        #self.fc1 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(1536, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        var_vector = torch.var(x, dim=(2, 3), keepdim=False)
        var_feat = self.variance_fc(var_vector)
        x = self.pool(x)
        x_flat = torch.flatten(x, 1)
        #x_combined = x_flat + var_feat
        x_combined = torch.cat((x_flat, var_feat), dim=1)  # Shape: [B, 1536]


        x = self.fc1(x_combined)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ---------- Parameters ----------
DATA_DIR = "./Dataset"
CATEGORIES = [
    "1.1 Water bottel whit cold water", "1.2 Water bottel whit rests of cold water", "1.3 Water bottel whit no water",
    "1.4 Water bottel whit body temp water", "1.5 Water bottel whit rest of body temp water",
    "2.6 Cottage Cheese before eating", "2.7 Cottage Cheese after eating",
    "3.8 Shower apartment fungus", "3.9 Shower apartment less fungus",
    "4.10 Basement humidity", "4.11 Basement humidity",
    "5.12 window 1", "5.13 window 2", "5.14 window 3", "5.15 window 4",
    "6.16 punctured window 1", "6.17 punctured window 2", "6.18 punctured window 3", "6.19 punctured window 4",
    "7.20 Bedrom window cold spot", "8.21 Cold spot wall", "9.22 Cold spots, old window"
]
BATCH_SIZE = 184
EPOCHS = 100
LEARNING_RATE = 0.0001

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------- Data ----------
dataset = HumidityDataset(DATA_DIR, CATEGORIES, transform=transform)
val_size = int(0.1 * len(dataset))
test_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------- Model Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HumidityCNN(num_classes=len(CATEGORIES)).to(device)

# ---------- Info ----------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")

# ---------- Loss / Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ---------- Training Loop ----------
loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()
    train_preds = []
    train_labels = []

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_train_acc = metrics.accuracy_score(train_labels, train_preds)

    # ---------- Validation ----------
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    epoch_val_acc = metrics.accuracy_score(val_labels, val_preds)
    loss_history.append(epoch_loss)
    train_acc_history.append(epoch_train_acc)
    val_acc_history.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")

torch.save(model.state_dict(), "humidity_model.pth")
print("✅ Model saved as humidity_model.pth")


# ---------- Final Validation Report ----------
val_conf_matrix = metrics.confusion_matrix(val_labels, val_preds)
val_accuracy = metrics.accuracy_score(val_labels, val_preds)
val_report = metrics.classification_report(val_labels, val_preds, target_names=CATEGORIES)

plt.figure(figsize=(10, 7))
sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Validation Confusion Matrix')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Confusion_matrix_val.jpg")
plt.show()

# ---------- Plot training curves ----------
plt.figure(figsize=(10, 6))
epochs_range = range(1, EPOCHS + 1)
plt.plot(epochs_range, loss_history, label='Loss')
plt.plot(epochs_range, train_acc_history, label='Train Accuracy')
plt.plot(epochs_range, val_acc_history, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_accuracy_plot.jpg")
plt.show()

# ---------- Test Evaluation ----------
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = metrics.accuracy_score(test_labels, test_preds)
test_conf_matrix = metrics.confusion_matrix(test_labels, test_preds)
test_report = metrics.classification_report(test_labels, test_preds, target_names=CATEGORIES)

print(f"\n✅ Final Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(test_report)

plt.figure(figsize=(10, 7))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Confusion Matrix')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Confusion_matrix_test.jpg")
plt.show()
