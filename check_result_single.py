import numpy as np

# Load the .npy file
file_path = './cri_predictions/single/cri_single_rgb+ss_10epoch.npy'  # Replace with your file path

data = np.load(file_path, allow_pickle=True)
data = data.item()

sorted_data = dict(sorted(data.items()))
for k, v in sorted_data.items():
    print(k, v)