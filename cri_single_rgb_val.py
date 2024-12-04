"""
This code is a failure
"""
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import glob

# Dataset Class
class CRIDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.image_paths = []
        self.image_labels = []
        for label, folder in labels.items():
            folder_path = os.path.join(img_dir, folder)
            images = glob.glob(os.path.join(folder_path, "*.png"))
            self.image_paths.extend(images)
            self.image_labels.extend([label] * len(images))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Preparation
def prepare_data(data_dir, batch_size):
    labels = {0: "cri_0", 1: "cri_1", 2: "cri_2", 3: "cri_3", 4: "cri_4"}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CRIDataset(data_dir, labels, transform)
    
    # Handle class imbalance with WeightedRandomSampler
    class_weights = compute_class_weight('balanced', classes=np.unique(dataset.image_labels), y=dataset.image_labels)
    sample_weights = [class_weights[label] for label in dataset.image_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader, len(labels)

# Define the Model
class CRIClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CRIClassifier, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# Loss Function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

# Training Loop
def train_model(model, train_loader, num_epochs, device, learning_rate=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save Predictions
def save_predictions(model, test_dir, output_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_images = glob.glob(os.path.join(test_dir, "*.png"))
    results = []
    with torch.no_grad():
        for img_path in test_images:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            results.append(predicted.item())
    np.save(output_path, np.array(results))
    print(f"Predictions saved to {output_path}")

# Main Execution
if __name__ == "__main__":
    train_dir = "./student_dataset/train/current_image"
    test_dir = "./student_dataset/student_test/current_image"
    output_path = "./predictions.npy"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_epochs = 10
    
    # Prepare Data
    train_loader, num_classes = prepare_data(train_dir, batch_size)
    
    # Initialize Model
    model = CRIClassifier(num_classes)
    
    # Train Model
    train_model(model, train_loader, num_epochs, device)
    
    # Save Predictions
    save_predictions(model, test_dir, output_path, device)
