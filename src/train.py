# This script trains a model to recognize surf maneuvers from sequences of video frames.
# It expects a directory structure created by maneuver_sequencing.py, where each ride
# has been split into 2-second sequences of frames and labeled with maneuver IDs.

# The model uses a 3D CNN (R3D-18) to learn spatio-temporal features from video sequences.
# It can run in two modes:
#  > dev: Uses grayscale images and fewer epochs for faster development
#  > prod: Uses RGB images and more epochs for better performance

# Usage:
# src $ python train.py --mode dev

print('>  Importing modules...')
import argparse, pytz, time, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from collections import Counter

from checkpoints import get_available_checkpoints, load_checkpoint, save_checkpoint
from dataset import SurfManeuverDataset
from model import SurfManeuverModel
from model_logging import write_training_log
from utils import (
    distribution_outdated, 
    format_time,
    load_maneuver_taxonomy, 
    load_class_distribution, 
    save_class_distribution, 
    setDevice
)

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Train a surf maneuver detection model.')
parser.add_argument('--mode', choices=['prod', 'dev'], default='dev', help='Set the application mode (prod or dev).')
parser.add_argument('--focal_loss', action='store_true', help='Use Focal Loss instead of weighted Cross Entropy.')
parser.add_argument('--weight_method', choices=['inverse', 'effective', 'sqrt', 'manual', 'balanced', 'power75', 'none'], default='none', 
                   help='Method for calculating class weights. Use "none" for no weighting.')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma parameter for Focal Loss (if used).')
parser.add_argument('--unfreeze_backbone', action='store_true', 
                   help='Unfreeze the backbone of the model to train all parameters. Default is False (backbone frozen).')
parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler. Disabled by default for initial training.')
args = parser.parse_args()
mode = args.mode
use_focal_loss = args.focal_loss
weight_method = args.weight_method
focal_gamma = args.gamma
freeze_backbone = not args.unfreeze_backbone  # Invert the flag to get freeze_backbone
use_scheduler = args.use_scheduler

# Model training choices
TRAIN_FROM_SCRATCH = 1
RESUME_FROM_CHECKPOINT = 2

# Ask user if they want to resume from checkpoint
print("\nDo you want to:")
print("1. Train a new model from scratch")
print("2. Resume training from a checkpoint")
while True:
    try:
        model_choice = int(input("\nEnter your choice (1 or 2): "))
        if model_choice in [TRAIN_FROM_SCRATCH, RESUME_FROM_CHECKPOINT]:
            break
        print("Please enter 1 or 2")
    except ValueError:
        print("Please enter a valid number")

# Initialize variables
start_epoch = 0
timestamp = None
total_elapsed_time = 0.0
epoch_losses, epoch_times = [], []
is_old_format = False

# Hyperparameters, defaults for dev mode if not loaded from checkpoint
if model_choice == TRAIN_FROM_SCRATCH:
    batch_size = 1
    learning_rate = 0.01
    num_epochs = 1
    if mode == 'prod':
        batch_size = 8
        learning_rate = 0.0025
        num_epochs = 20

    # Generate new timestamp for fresh training
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    timestamp = now.strftime("%Y%m%d_%H%M")

if model_choice == RESUME_FROM_CHECKPOINT:
    # List available checkpoints
    checkpoints = get_available_checkpoints()
    if not checkpoints:
        print("Error: No checkpoints found. Please train a new model first.")
        sys.exit(1)
    
    print("\nAvailable checkpoints:")
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. {cp['filename']} (Epoch {cp['epoch']})")
    
    # Get user's checkpoint choice
    while True:
        try:
            cp_choice = int(input("\nEnter the number of the checkpoint to resume from: "))
            if 1 <= cp_choice <= len(checkpoints):
                selected_cp = checkpoints[cp_choice - 1]
                break
            print(f"Please enter a number between 1 and {len(checkpoints)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Load checkpoint data
    checkpoint_path = os.path.join("../models", selected_cp['filename'])
    print(f"\nLoading checkpoint: {checkpoint_path}")
    try:
        model_state, optimizer_state, start_epoch, timestamp, training_config, training_history = load_checkpoint(checkpoint_path)
        
        # Load training history if it exists
        if training_history:
            epoch_losses = training_history['epoch_losses']
            epoch_times = training_history['epoch_times']
            total_elapsed_time = training_history['total_elapsed_time']
        else:
            is_old_format = True
        
        # Apply saved training config if available
        if training_config:
            print("Using training configuration from checkpoint")
            mode = training_config['mode']
            batch_size = training_config['batch_size']
            learning_rate = training_config['learning_rate']
            num_epochs = training_config['num_epochs']
            use_focal_loss = training_config['use_focal_loss']
            weight_method = training_config['weight_method']
            focal_gamma = training_config['focal_gamma']
            freeze_backbone = training_config['freeze_backbone']
            use_scheduler = training_config['use_scheduler']
        
        print(f"Resuming from epoch {start_epoch}")
        print(f"Previous training time: {total_elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

# Set device to GPU if available, otherwise use CPU
device = setDevice()

# Data preparation
print('>  Loading dataset...')
dataset = SurfManeuverDataset(root_dir="../data/heats", transform=None, mode=mode)
print('>  Creating dataloader...')

# Calculate or load class distribution
print('>  Checking class distribution...')
if distribution_outdated():
    print('  >  Recalculating class distribution...')
    class_distribution = Counter()
    for _, label in dataset:
        class_distribution[label] += 1
    save_class_distribution(class_distribution)
else:
    print('  >  Loading cached class distribution...')
    class_distribution = load_class_distribution()

total_samples = sum(class_distribution.values())
num_classes = max(class_distribution.keys()) + 1

# Load maneuver names from taxonomy
maneuver_names = load_maneuver_taxonomy()

# Calculate the maximum number of digits in any count and add one for padding
max_count_width = max(len(str(count)) for count in class_distribution.values()) + 1

print('>  Class distribution:')
for class_id in range(num_classes):
    count = class_distribution.get(class_id, 0)
    percentage = (count / total_samples) * 100
    name = maneuver_names.get(class_id, f"Unknown-{class_id}")
    print(f'  > Class {class_id} - {name}: {count:>{max_count_width}} samples ({percentage:>5.2f}%)')

# Calculate class weights based on distribution
class_counts = torch.zeros(num_classes)
for class_id in range(num_classes):
    class_counts[class_id] = class_distribution.get(class_id, 1)  # Default to 1 if class not found

# Different methods to calculate weights
if weight_method == 'none':
    # No weighting
    class_weights = None
else:
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
    elif weight_method == 'power75':
        # p^-0.75 power weighting (between inverse and sqrt)
        class_frequencies = class_counts / sum(class_counts)
        class_weights = 1.0 / torch.pow(class_frequencies + 1e-5, 0.75)
        class_weights = class_weights / class_weights.sum() * num_classes
    elif weight_method == 'balanced':
        # New balanced approach - more conservative weighting to avoid over-penalizing common classes
        class_frequencies = class_counts / sum(class_counts)
        # Apply cube root for even more moderate scaling than sqrt
        class_weights = 1.0 / torch.pow(class_frequencies + 1e-5, 1/3)
        class_weights = class_weights / class_weights.sum() * num_classes
    else:  # 'manual'
        # Manual weighting: less weight for "No maneuver", more for others
        class_weights = torch.ones(num_classes) 
        class_weights[0] = 0.75  # Reduced penalty for "No maneuver" (was 0.6)
        # More moderate boost for actual maneuvers
        for i in range(1, num_classes):
            if i in [3, 5]:  # 360 and Air (rarest classes)
                class_weights[i] = 1.5  
            else:
                class_weights[i] = 1.25  # Other maneuvers

    print('>  Class weights:')
    for class_id in range(num_classes):
        print(f'  > Class {class_id}: {class_weights[class_id]:.4f}')

    class_weights = class_weights.to(device)

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
model = SurfManeuverModel(mode=mode, freeze_backbone=freeze_backbone)
model = model.to(device)

# If resuming, validate and load the model state
if model_choice == RESUME_FROM_CHECKPOINT:
    # Validate model architecture by comparing state dict keys
    current_model_state = model.state_dict()
    if set(current_model_state.keys()) != set(model_state.keys()):
        raise ValueError("Model architecture in checkpoint does not match current model")
    model.load_state_dict(model_state)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if model_choice == RESUME_FROM_CHECKPOINT and optimizer_state is not None:
    optimizer.load_state_dict(optimizer_state)

# Define Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None):  # Reduced gamma from 2.0 to 1.0 for less aggressive focus
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Create loss function with class weights
if use_focal_loss:
    criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
elif weight_method != 'none':
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

# Add learning rate scheduler for better convergence
scheduler = None
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )

# Print final configuration
print('\n>  Training configuration:')
print('>    Mode:', mode)
print('>    Batch size:', batch_size)
print('>    Learning rate:', learning_rate)
print('>    Number of epochs:', num_epochs)
print('>    Loss function:', 'Focal Loss' if use_focal_loss else 'Cross Entropy Loss')
print('>    Class weighting:', weight_method)
if use_focal_loss:
    print('>    Focal loss gamma:', focal_gamma)
print('>    Backbone frozen:', freeze_backbone)
print('>    Learning rate scheduler:', 'Enabled' if use_scheduler else 'Disabled')
print()

# Training loop
print('>  Starting training...')
epoch_start_time = time.time()  # Track time for this epoch
for epoch in range(start_epoch, num_epochs):
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
        
        # Print loss every N batches for progress feedback
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"    >  Batch {batch_idx + 1}/{total_batches}, Loss: {loss_value:.4f}")

    # Calculate and track epoch loss
    epoch_loss = running_loss / total_batches
    epoch_losses.append(epoch_loss)
    
    # Update learning rate based on loss
    if scheduler is not None:
        scheduler.step(epoch_loss)
    
    # Calculate time for this epoch and update total
    epoch_duration = time.time() - epoch_start_time
    epoch_times.append(epoch_duration)  # Store this epoch's time
    total_elapsed_time += epoch_duration
    
    # Print average loss and time taken for the epoch
    print(f"    >  Epoch [{epoch+1}/{num_epochs}] completed in {format_time(epoch_duration)}. Average Loss: {epoch_loss:.4f}")    
    print(f"    >  Current learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"    >  Total training time so far: {format_time(total_elapsed_time)}")

    # Save model checkpoint for each epoch except the last one
    if epoch < num_epochs - 1:  # Skip the last epoch as it will be saved as the final model
        training_config = {
            'mode': mode,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'use_focal_loss': use_focal_loss,
            'weight_method': weight_method,
            'focal_gamma': focal_gamma,
            'freeze_backbone': freeze_backbone,
            'use_scheduler': use_scheduler
        }
        training_history = {
            'epoch_losses': epoch_losses,
            'epoch_times': epoch_times,
            'total_elapsed_time': total_elapsed_time
        }
        checkpoint_path = save_checkpoint(model, optimizer, epoch, timestamp, class_distribution, training_config, training_history=training_history)
        print(f"    >  Model checkpoint saved: {checkpoint_path}")
    
    # Reset epoch timer for next epoch
    epoch_start_time = time.time()

print("Training complete.")

# Save final model
training_config = {
    'mode': mode,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs,
    'use_focal_loss': use_focal_loss,
    'weight_method': weight_method,
    'focal_gamma': focal_gamma,
    'freeze_backbone': freeze_backbone,
    'use_scheduler': use_scheduler
}
training_history = {
    'epoch_losses': epoch_losses,
    'epoch_times': epoch_times,
    'total_elapsed_time': total_elapsed_time
}
model_filename = save_checkpoint(model, optimizer, num_epochs - 1, timestamp, class_distribution, training_config, is_final=True, training_history=training_history)
print(f"Final model saved: {model_filename}")

# Write training log
log_filename = f"../logs/training_{timestamp}.log"
write_training_log(
    log_filename=log_filename,
    timestamp=timestamp,
    mode=mode,
    batch_size=batch_size,
    learning_rate=learning_rate,
    num_epochs=num_epochs,
    use_focal_loss=use_focal_loss,
    weight_method=weight_method,
    focal_gamma=focal_gamma,
    freeze_backbone=freeze_backbone,
    class_distribution=class_distribution,
    maneuver_names=maneuver_names,
    total_elapsed_time=total_elapsed_time,
    epoch_losses=epoch_losses,
    epoch_times=epoch_times,
    final_lr=optimizer.param_groups[0]['lr'],
    is_old_format=is_old_format
)

print(f"Training log saved to: {log_filename}")
print(f"Total training time: {format_time(total_elapsed_time)}")