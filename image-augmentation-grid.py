import os
import random
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image

# Directory containing training images
image_dir = './training/PlantVillage/train'

# Define the transformation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
])

# Get a list of all image paths recursively with labels
image_files = []
labels = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            image_files.append(os.path.join(root, file))
            labels.append(os.path.basename(root))

# Check if there are enough images to sample
if len(image_files) < 9:
    raise ValueError(f"Not enough images in {image_dir}. Found {len(image_files)}, but 9 are required.")

# Randomly select 9 images and their labels
selected_indices = random.sample(range(len(image_files)), 9)
selected_images = [image_files[i] for i in selected_indices]
selected_labels = [labels[i] for i in selected_indices]

# Create a 16:9 grid
fig, axes = plt.subplots(3, 3, figsize=(16, 9))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for ax, img_path, label in zip(axes.flatten(), selected_images, selected_labels):
    # Load the image as a PIL image for proper transformation
    img = Image.open(img_path)
    transformed_img = train_transform(img).permute(1, 2, 0).numpy()
    transformed_img = (transformed_img * 0.229 + 0.485).clip(0, 1)  # Denormalize
    ax.imshow(transformed_img)
    ax.set_title(label, fontsize=10)
    ax.axis('off')

plt.show()
