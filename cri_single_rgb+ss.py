import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, models
import numpy as np





class CRIDataset(Dataset):
    def __init__(self, rgb_dir, seg_dir, labels=None, transform=None):
        self.rgb_dir = rgb_dir
        self.seg_dir = seg_dir
        self.labels = labels
        self.transform = transform
        self.files = sorted(os.listdir(rgb_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.files[idx])
        seg_path = os.path.join(self.seg_dir, self.files[idx])
        
        rgb_image = Image.open(rgb_path).convert("RGB")
        seg_image = Image.open(seg_path).convert("RGB")
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            seg_image = self.transform(seg_image)

        data = {"rgb": rgb_image, "seg": seg_image}
        if self.labels:
            label = self.labels[idx]
            return data, label
        return data




# Paths
TRAIN_RGB_DIR = "./student_dataset/train/current_image"
TRAIN_SEG_DIR = "./results_train/segmentation/filtered"
#CLASS_WEIGHTS = [300, 100, 100, 80, 100]

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.3

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Data
datasets = []
labels = []
for label in range(5):
    rgb_dir = os.path.join(TRAIN_RGB_DIR, f"cri_{label}")
    seg_dir = os.path.join(TRAIN_SEG_DIR, f"cri_{label}")
    files = os.listdir(rgb_dir)
    datasets.append(CRIDataset(rgb_dir, seg_dir, [label] * len(files), transform))
    labels.extend([label] * len(files))

# Combine datasets
combined_datasets = torch.utils.data.ConcatDataset(datasets)

"""
# Weighted Random Sampler
weights = [1.0 / CLASS_WEIGHTS[label] for label in labels]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
"""

# Split dataset
val_size = int(VAL_SPLIT * len(combined_datasets))
train_size = len(combined_datasets) - val_size
train_dataset, val_dataset = random_split(combined_datasets, [train_size, val_size]) # [476, 204]


############
train_labels = []
for idx in range(len(train_dataset)):
    _, label = train_dataset[idx]
    train_labels.append(label)
    

from collections import Counter
label_counts = Counter(train_labels)
total_samples = len(train_labels)
class_weights = {label: total_samples / count for label, count in label_counts.items()}
# Assign weights to each sample based on its label
sample_weights = [class_weights[label] for label in train_labels]


sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
############


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
class CRIModel(nn.Module):
    def __init__(self):
        super(CRIModel, self).__init__()
        self.rgb_model = models.resnet18(pretrained=True)
        self.seg_model = models.resnet18(pretrained=True)
        self.fc = nn.Linear(2 * 1000, 5)  # 5 classes

    def forward(self, rgb, seg):
        rgb_features = self.rgb_model(rgb)
        seg_features = self.seg_model(seg)
        combined = torch.cat((rgb_features, seg_features), dim=1)
        return self.fc(combined)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRIModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        data, labels = batch
        rgb = data["rgb"].to(device)
        seg = data["seg"].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb, seg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            data, labels = batch
            rgb = data["rgb"].to(device)
            seg = data["seg"].to(device)
            labels = labels.to(device)
            
            outputs = model(rgb, seg)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")


# Paths
TEST_RGB_DIR = "./student_dataset/student_test/current_image"
TEST_SEG_DIR = "./results/segmentation/filtered"
OUTPUT_FILE = "cri_predictions.npy"

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Model
model.eval()

# Prediction
predictions = {}
for file in os.listdir(TEST_RGB_DIR):
    rgb_image = transform(Image.open(os.path.join(TEST_RGB_DIR, file)).convert("RGB"))
    seg_image = transform(Image.open(os.path.join(TEST_SEG_DIR, file)).convert("RGB"))
    rgb_image, seg_image = rgb_image.unsqueeze(0), seg_image.unsqueeze(0)
    rgb_image = rgb_image.to(device)
    seg_image = seg_image.to(device)
    output = model(rgb_image, seg_image)
    _, predicted = torch.max(output, 1)
    predictions[file] = predicted.item()

# Save Predictions
sorted_predictions = dict(sorted(predictions.items()))
np.save(OUTPUT_FILE, sorted_predictions)