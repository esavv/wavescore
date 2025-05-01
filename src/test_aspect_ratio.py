"""
Test script to compare the old square resize transformation with the new aspect-ratio preserving transformation.
This script:
1. Takes a sample frame
2. Applies both transformations
3. Saves the results to visualize the difference
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from utils import preserve_aspect_ratio_transform

# Find a sample frame
frame_path = None
for root, dirs, files in os.walk('../data/heats'):
    for file in files:
        if file.endswith('.jpg'):
            frame_path = os.path.join(root, file)
            break
    if frame_path:
        break

if not frame_path:
    print("Error: No sample frames found. Please check the data directory.")
    exit(1)

print(f"Using sample frame: {frame_path}")

# Load the image
image = Image.open(frame_path)
original_size = image.size
print(f"Original image size: {original_size}")

# Create a figure to display the results
plt.figure(figsize=(15, 5))

# Display original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title(f"Original ({original_size[0]}x{original_size[1]})")

# Apply old transformation (square resize)
old_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
old_transformed = old_transform(image)
old_transformed_img = transforms.ToPILImage()(old_transformed)

# Display old transformation result
plt.subplot(1, 3, 2)
plt.imshow(old_transformed_img)
plt.title("Old Transform (Distorted)")

# Apply new transformation (aspect ratio preserving)
new_transformed = preserve_aspect_ratio_transform(image, target_size=224)
new_transformed_img = transforms.ToPILImage()(new_transformed)

# Display new transformation result
plt.subplot(1, 3, 3)
plt.imshow(new_transformed_img)
plt.title("New Transform (Preserved Ratio)")

# Save the comparison
plt.tight_layout()
plt.savefig('../aspect_ratio_comparison.png')
print(f"Comparison saved to: '../aspect_ratio_comparison.png'")
plt.close()

print("Test complete. Check the output image to see the transformation difference.") 