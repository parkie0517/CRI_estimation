"""
This file is used to filter semantic segmentation results.

"""
import os
import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar


colors = { 
    'null': [  0,   0,   0], # null
    'road': [128, 64, 128], #road
    'sidewalk': [244, 35, 232], #sidewalk
    'building':[70, 70, 70], #building
    'wall':[102, 102, 156],#wall
    'fence':[190, 153, 153],#fence
    'pole':[153, 153, 153],#pole
    'traffic_light':[250, 170, 30],#traffic light
    'traffiic_sign':[220, 220, 0],#traffiic sign
    'vegetation':[107, 142, 35],  # vegetation dark green
    'terrain':[152, 251, 152],  # terrain bright green
    'sky':[0, 130, 180],#sky
    'person':[220, 20, 60], #person
    'rider':[255, 0, 0], # rider
    'car':[0, 0, 142],
    'truck':[0, 0, 70],
    'bus':[0, 60, 100],
    'train':[0, 80, 100],
    'motorcycle':[0, 0, 230], # motorcycle
    'bicycle':[119, 11, 32], # bicycle
}

# Define allowed categories
allowed_colors = [
    colors['road'],
    colors['sidewalk'],
    colors['person'],
    colors['rider'],
    colors['motorcycle'],
    colors['bicycle']
]

# load images from the directory
src_path = '/home/vilab/ssd1tb/hj_ME455/Term_Project/results/segmentation/color'
dst_path = '/home/vilab/ssd1tb/hj_ME455/Term_Project/results/segmentation/filtered'


# Create destination path if it doesn't exist
os.makedirs(dst_path, exist_ok=True)


# Get list of image paths
image_list = [os.path.join(src_path, fname) for fname in os.listdir(src_path) if fname.endswith(('.png'))]


# Loop through the images with a progress bar
for image_path in tqdm(image_list, desc="Processing images", unit="image"):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Get image dimensions
    height, width, _ = image.shape

    # Fill a rectangle with [0, 0, 0]
    cv2.rectangle(image, (448, height - 50), (2048 - 448, height), (0, 0, 0), -1)

    # Filter pixels based on allowed categories
    filtered_image = np.zeros_like(image)  # Initialize a blank image
    
    for color in allowed_colors:
        # Create a mask for the current color
        mask = np.all(image == color, axis=-1)
        filtered_image[mask] = color

    # Convert the filtered image back to BGR for saving
    filtered_image_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR)

    # Save the filtered image to the destination path
    output_path = os.path.join(dst_path, os.path.basename(image_path))
    cv2.imwrite(output_path, filtered_image_bgr)
    
    breakpoint()