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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRIModel().to(device)

"""
# Save the model checkpoint after each epoch
checkpoint = {
    'model_state_dict': model.state_dict()
}
torch.save(checkpoint, f'checkpoint_10epoch.pth')
print("Done saving")
"""


# Load the model checkpoint
checkpoint = torch.load('checkpoint_10epoch.pth')  # Replace with your checkpoint file
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Set the model to evaluation mode if you are using it for inference
model.eval()  # Or model.train() if you want to resume training


# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



# After training or during evaluation
# Example: Generating CAM for a sample test image

image_name = "./cam_results/bremen_000193_000020_leftImg8bit.png"

sample_rgb_path = "./student_dataset/student_test/current_image/"+ image_name  # Update with actual path
sample_image = Image.open(sample_rgb_path).convert("RGB")
sample_image_tensor = transform(sample_image)

# Generate and save CAM for the sample image
generate_and_save_cam(model, sample_image_tensor, save_path=f"cam_output_{image_name}")
