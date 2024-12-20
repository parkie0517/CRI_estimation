import numpy as np

# Load the .npy file
file_path = './cri_predictions/sequential/cri_sequential_rgb_5epoch.npy'  # Replace with your file path

data = np.load(file_path, allow_pickle=True)
data = data.item()

sorted_data = dict(sorted(data.items()))
for k, v in sorted_data.items():
    print(k, v)