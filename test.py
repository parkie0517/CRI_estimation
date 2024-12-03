import numpy as np

# Load the .npy file
file_path = 'val_sanity.npy'  # Replace with your file path

data = np.load(file_path, allow_pickle=True)

data_dict = data.item()
print(f"The data contains {len(data_dict)} items.")
print("Sample items from the dictionary:")

# Print a few sample key-value pairs
for key, value in list(data_dict.items())[:10]:  # Change 10 to view more samples
    print(f"{key}: {value}")