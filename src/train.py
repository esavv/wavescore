# This script trains a model to recognize surf maneuvers from sequences of video frames.
# It expects a directory structure created by maneuver_sequencing.py, where each ride
# has been split into 2-second sequences of frames and labeled with maneuver IDs.

# The model combines a CNN (for frame feature extraction) with an LSTM (to learn
# temporal relationships between frames). It can run in two modes:
#  > dev: Uses grayscale images and fewer epochs for faster development
#  > prod: Uses RGB images and more epochs for better performance

# Usage:
# src $ python train.py --mode dev

print('>  Importing modules...')
import argparse, pytz, time, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SurfManeuverDataset
from model import SurfManeuverModel

# Define Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Train a surf maneuver detection model.')
parser.add_argument('--mode', choices=['prod', 'dev'], default='dev', help='Set the application mode (prod or dev).')
parser.add_argument('--focal_loss', action='store_true', help='Use Focal Loss instead of weighted Cross Entropy.')
parser.add_argument('--weight_method', choices=['inverse', 'effective', 'sqrt', 'manual'], default='sqrt', 
                   help='Method for calculating class weights.')
parser.add_argument('--visualize', action='store_true', help='Visualize class distribution and training progress.')
args = parser.parse_args()
mode = args.mode
use_focal_loss = args.focal_loss
weight_method = args.weight_method
visualize = args.visualize

# Hyperparameters, defaults for dev mode
batch_size = 1
learning_rate = 0.01
num_epochs = 1
if mode == 'prod':
    batch_size = 3        # 4
    learning_rate = 0.001 # 0.001
    num_epochs = 5        # 10
print('>  Setting hyperparameters... (mode is: ' + mode + ')')

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
print('>  Creating dataset...')
dataset = SurfManeuverDataset(root_dir="../data/heats", transform=None, mode=mode)
print('>  Creating dataloader...')

# Calculate class distribution
class_distribution = Counter()
print('>  Calculating class distribution...')
for _, label in dataset:
    class_distribution[label] += 1

total_samples = sum(class_distribution.values())
num_classes = max(class_distribution.keys()) + 1

print('>  Class distribution:')
for class_id in range(num_classes):
    count = class_distribution.get(class_id, 0)
    percentage = (count / total_samples) * 100
    print(f'  > Class {class_id}: {count} samples ({percentage:.2f}%)')

# Calculate class weights based on distribution
class_counts = torch.zeros(num_classes)
for class_id in range(num_classes):
    class_counts[class_id] = class_distribution.get(class_id, 1)  # Default to 1 if class not found

# Different methods to calculate weights
if weight_method == 'inverse':
    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-5)  # Small epsilon to avoid division by zero
    class_weights = class_weights / class_weights.sum() * num_classes
elif weight_method == 'effective':
    # Effective number of samples (Cui et al., 2019)
    beta = 0.9999
    effective_num = 1.0 - torch.pow(beta, class_counts)
    class_weights = (1.0 - beta) / (effective_num + 1e-5)
    class_weights = class_weights / class_weights.sum() * num_classes
elif weight_method == 'sqrt':
    # Square root of inverse frequency (more moderate)
    class_frequencies = class_counts / sum(class_counts)
    class_weights = 1.0 / torch.sqrt(class_frequencies + 1e-5)
    class_weights = class_weights / class_weights.sum() * num_classes
else:  # 'manual'
    # Manual weighting: less weight for "No maneuver", more for others
    class_weights = torch.ones(num_classes)
    class_weights[0] = 0.6  # Reduce weight for "No maneuver"
    for i in range(1, num_classes):
        class_weights[i] = 1.3  # Increase weight for actual maneuvers

print('>  Class weights:')
for class_id in range(num_classes):
    print(f'  > Class {class_id}: {class_weights[class_id]:.4f}')

class_weights = class_weights.to(device)

# Visualize class distribution if requested
if visualize:
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), [class_distribution.get(i, 0) for i in range(num_classes)])
    plt.title('Class Distribution')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.savefig(f'../logs/class_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda x: (
        torch.stack([item[0] for item in x]),  # Only one torch.stack here to batch tensors
        torch.tensor([item[1] for item in x])
    )
)
total_batches = len(dataloader)
start_time = time.time()  # Track the start time of training

# Model, loss function, and optimizer
print('>  Defining the model...')
model = SurfManeuverModel(mode=mode)
model = model.to(device)

# Create loss function with class weights
if use_focal_loss:
    print('>  Using Focal Loss with class weights')
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
else:
    print('>  Using Cross Entropy Loss with class weights')
    criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to track metrics
epoch_losses = []
batch_losses = []

# Training loop
print('>  Starting training...')
for epoch in range(num_epochs):
    print(f'  >  Epoch {epoch+1}/{num_epochs}')
    running_loss = 0.0
    
    for batch_idx, (frames, labels) in enumerate(dataloader):
        # Print batch progress
        print(f"    >  Processing batch {batch_idx + 1}/{total_batches}")

        # Move data to the correct device
        frames, labels = frames.to(device), labels.to(device)
        
        # Forward pass
        outputs, _ = model(frames)  # Discard the None/hidden state from the output
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        loss_value = loss.detach().item()
        running_loss += loss_value
        batch_losses.append(loss_value)
        
        # Print loss every N batches for progress feedback
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"    >  Batch {batch_idx + 1}/{total_batches}, Loss: {loss_value:.4f}")

    # Calculate and track epoch loss
    epoch_loss = running_loss / total_batches
    epoch_losses.append(epoch_loss)
    
    # Print average loss and time taken for the epoch
    epoch_duration = time.time() - start_time
    print(f"    >  Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds. Average Loss: {epoch_loss:.4f}")    

    # Save model checkpoint for each epoch
    if epoch > 0 and epoch % 1 == 0:  # Save every epoch
        checkpoint_path = f"../models/surf_maneuver_model_checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"    >  Model checkpoint saved: {checkpoint_path}")

# Visualize training progress if requested
if visualize:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(batch_losses)
    plt.title('Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f'../logs/training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

print("Training complete.")

est = pytz.timezone('US/Eastern')
now = datetime.now(est)
timestamp = now.strftime("%Y%m%d_%H%M")

# Save final model
model_filename = f"../models/surf_maneuver_model_{timestamp}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Final model saved: {model_filename}")

# Save training configuration
config = {
    'mode': mode,
    'use_focal_loss': use_focal_loss,
    'weight_method': weight_method,
    'class_weights': class_weights.cpu().numpy().tolist(),
    'class_distribution': {str(k): v for k, v in class_distribution.items()},
    'epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'final_loss': epoch_losses[-1]
}

# Save configuration to a log file
import json
with open(f"../logs/training_config_{timestamp}.json", 'w') as f:
    json.dump(config, f, indent=2)

print(f"Training configuration saved to: ../logs/training_config_{timestamp}.json")
