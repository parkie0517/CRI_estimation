import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import cv2




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
#val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
class CRIModel(nn.Module):
    def __init__(self):
        super(CRIModel, self).__init__()
        self.rgb_model = models.resnet18(pretrained=True)
        self.seg_model = models.resnet18(pretrained=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2 * 1000, 512)  # First hidden layer
        self.bn1 = nn.BatchNorm1d(512)       # Batch normalization
        self.fc2 = nn.Linear(512, 5)         # Output layer for 5 classes

    def forward(self, rgb, seg):
        rgb_features = self.rgb_model(rgb)
        seg_features = self.seg_model(seg)
        
        # Concatenate features
        combined = torch.cat((rgb_features, seg_features), dim=1)
        
        # Pass through the hidden layer, BN, and activation
        x = self.fc1(combined)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        
        # Pass through the final output layer
        x = self.fc2(x)
        return x 
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRIModel().to(device)


"""
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
    
    
# Save the model checkpoint after each epoch
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': running_loss / len(train_loader)
}
torch.save(checkpoint, f'checkpoint_{EPOCHS}epoch.pth')
print("Done saving")

"""

# Load saved Model

# Load the model checkpoint
checkpoint = torch.load('checkpoint_epoch_1.pth')  # Replace with your checkpoint file
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode if you are using it for inference
model.eval()  # Or model.train() if you want to resume training

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



# Function to generate and save CAM
def generate_and_save_cam(model, image_tensor, save_path, target_class=None):
    
    #Generate and save the Class Activation Map (CAM) for the given image tensor.
    
    model.eval()
    
    # Extract the features and classifier weights
    features_blobs = []
    def hook_fn(module, input, output):
        features_blobs.append(output)
    
    # Hook the last convolutional layer of the RGB model
    final_layer = model.rgb_model.layer4
    hook = final_layer.register_forward_hook(hook_fn)
    
    # Perform a forward pass
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device), image_tensor.unsqueeze(0).to(device))
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
    
    # Get weights of the classifier
    params = list(model.fc2.parameters())
    weight_softmax = params[0].cpu().detach().numpy()
    
    # Get the last convolutional features
    features = features_blobs[0].squeeze(0).cpu().detach().numpy()
    
    # Calculate the CAM
    cam = weight_softmax[target_class].dot(features.reshape(features.shape[0], -1))
    cam = cam.reshape(features.shape[1:])
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0, 1]
    
    # Overlay the CAM on the original image
    original_image = image_tensor.permute(1, 2, 0).numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    overlayed_image = heatmap * 0.4 + original_image
    
    # Save and display the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Class Activation Map")
    plt.imshow(overlayed_image)
    plt.axis("off")
    
    plt.savefig(save_path)
    plt.show()
    
    # Remove the hook
    hook.remove()

# After training or during evaluation
# Example: Generating CAM for a sample test image
sample_rgb_path = "./student_dataset/student_test/current_image/darmstadt_000022_000020_leftImg8bit.png"  # Update with actual path
sample_image = Image.open(sample_rgb_path).convert("RGB")
sample_image_tensor = transform(sample_image)

# Generate and save CAM for the sample image
generate_and_save_cam(model, sample_image_tensor, save_path="cam_output.png")
