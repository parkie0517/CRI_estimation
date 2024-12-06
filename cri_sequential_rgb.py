import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# Constants
DATA_DIR = "./student_dataset/train"
TEST_DIR = "./student_dataset/student_test"
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 3

# Dataset class
class CRIDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        self.samples = []
        if self.is_train:
            self._prepare_train_samples()
        else:
            self._prepare_test_samples()

    def _prepare_train_samples(self):
        
        for cri in range(NUM_CLASSES):
            
            current_dir = os.path.join(self.data_dir, f"current_image/cri_{cri}")
            past_dir = os.path.join(self.data_dir, f"past_image/cri_{cri}")
            for file in os.listdir(current_dir):
                if file.endswith(".png"):
                    
                    splitted_file = file.split('_')
                    past_images = []
                    for i in range(0, 20, 5):

                        past_images_dir = os.path.join(past_dir, f"{splitted_file[0]}_{splitted_file[1]}_0")
                        past_image = os.path.join(past_images_dir, f"{splitted_file[0]}_{splitted_file[1]}_{i:06d}_leftImg8bit.png")
                        past_images.append(past_image)

                        
                    self.samples.append((past_images, os.path.join(current_dir, file), cri))

    def _prepare_test_samples(self):
        current_dir = os.path.join(self.data_dir, "current_image")
        for file in os.listdir(current_dir):
            if file.endswith(".png"):
                
                splitted_file = file.split('_')
                past_images = []
                for i in range(0, 20, 5):
       
                
                    past_images_dir = os.path.join(self.data_dir, f"past_image/{splitted_file[0]}_{splitted_file[1]}_0")
                    past_image = os.path.join(past_images_dir, f"{splitted_file[0]}_{splitted_file[1]}_{i:06d}_leftImg8bit.png")
                    past_images.append(past_image)
                      
                self.samples.append((past_images, os.path.join(current_dir, file)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        if len(self.samples[idx]) == 2:
            past_paths, current_path = self.samples[idx]
            test_time = True
        else:
            past_paths, current_path, label = self.samples[idx]
            test_time = False
            
        
        past_images = [Image.open(img).convert("RGB") for img in past_paths]
        current_image = Image.open(current_path).convert("RGB")
        if self.transform:
            past_images = [self.transform(img) for img in past_images]
            current_image = self.transform(current_image)
        past_images = torch.stack(past_images)
        
        if test_time:
            return past_images, current_image, os.path.basename(current_path)
        else:
            return past_images, current_image, label


# Define Model
class CRIModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CRIModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, past_images, current_image):
        batch_size, seq_len, c, h, w = past_images.size()
        past_features = [self.backbone(past_images[:, i]) for i in range(seq_len)]
        past_features = torch.mean(torch.stack(past_features), dim=0)  # Temporal average
        current_features = self.backbone(current_image)
        combined = torch.cat((past_features, current_features), dim=1)
        return self.fc(combined)


# WeightedRandomSampler
def get_sampler(labels):
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, len(weights))


# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CRIDataset(DATA_DIR, is_train=True, transform=transform)
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.3, stratify=[sample[2] for sample in dataset.samples])
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_sampler = get_sampler([dataset.samples[idx][2] for idx in train_indices])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Training
model = CRIModel(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for past_images, current_image, labels in tqdm(train_loader, desc="Training Progress"):
        past_images, current_image, labels = past_images.to(DEVICE), current_image.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(past_images, current_image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_acc = correct / total

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for past_images, current_image, labels in val_loader:
            past_images, current_image, labels = past_images.to(DEVICE), current_image.to(DEVICE), labels.to(DEVICE)
            outputs = model(past_images, current_image)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = correct / total
    
    scheduler.step()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save predictions
test_dataset = CRIDataset(TEST_DIR, is_train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = {}
model.eval()
with torch.no_grad():
    for past_images, current_image, file_name in tqdm(test_loader, desc="Testing Progress"):
        past_images, current_image = past_images.to(DEVICE), current_image.to(DEVICE)
        outputs = model(past_images, current_image)
        _, predicted = outputs.max(1)

        file_name = file_name[0]
        predictions[file_name] = predicted.item()

# Save Predictions
sorted_predictions = dict(sorted(predictions.items()))

# change key name MAINZ
old_key = 'mainz_000002_000020_leftImg8bit.png'
new_key = 'mainz_000002_000062_leftImg8bit.png'
sorted_predictions[new_key] = sorted_predictions.pop(old_key)
np.save("./cri_predictions/sequential/cri_sequential_rgb_3epoch.npy", sorted_predictions)
