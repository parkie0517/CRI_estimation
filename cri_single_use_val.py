import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Train Dataset
class CRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        for label in range(NUM_CLASSES):
            dir_path = os.path.join(root_dir, f"cri_{label}")
            for file_name in os.listdir(dir_path):
                self.data.append(os.path.join(dir_path, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Paths
TRAIN_PATH = "./student_dataset/train/current_image"
TEST_PATH = "./student_dataset/student_test/current_image"

# Config
NUM_CLASSES = 5
CLASS_COUNTS = [300, 100, 100, 80, 100]
BATCH_SIZE = 128
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Weighted Sampler
class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=np.repeat(np.arange(NUM_CLASSES), CLASS_COUNTS))
sample_weights = [class_weights[label] for label in np.repeat(np.arange(NUM_CLASSES), CLASS_COUNTS)]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# Dataset splitting with 30% validation data for each label
dataset = CRIDataset(TRAIN_PATH, transform=train_transform)
label_indices = {label: [] for label in range(NUM_CLASSES)}

for idx, label in enumerate(dataset.labels):
    label_indices[label].append(idx)

train_indices, val_indices = [], []

for label in range(NUM_CLASSES):
    indices = label_indices[label]
    split = int(len(indices) * 0.3)
    val_indices.extend(indices[:split])
    train_indices.extend(indices[split:])

train_subset = torch.utils.data.Subset(dataset, train_indices)
val_subset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=4)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = FocalLoss(alpha=1, gamma=2)

# Training
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        val_loss, val_accuracy = validate_model()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step()

def validate_model():
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    return val_loss / len(val_loader), val_accuracy

# Test Predictions
def predict():
    model.eval()
    test_images_dir = "./student_dataset/student_test/current_image"
    test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(".png")]

    pred_dict = {}
    with torch.no_grad():
        for image_path in test_images:
            file_name_with_extension = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")
            image = test_transform(image).unsqueeze(0).to(device)
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            pred_dict[file_name_with_extension] = preds.item()

    # Sort predictions based on keys
    sorted_pred_dict = dict(sorted(pred_dict.items()))
    # Save predictions as an npy file
    np.save("cri_single.npy", np.array(sorted_pred_dict))
    print("Predictions saved to cri_single.npy")

if __name__ == "__main__":
    train_model()
    predict()
