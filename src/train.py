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
import argparse, pytz, time, os, re, sys
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
    def __init__(self, gamma=1.0, alpha=None):  # Reduced gamma from 2.0 to 1.0 for less aggressive focus
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def get_available_checkpoints():
    """List available checkpoints in the models directory."""
    checkpoint_dir = "../models"
    checkpoints = []
    
    # Get all .pth files with 'checkpoint' in the name
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth') and 'checkpoint' in filename:
            # Extract timestamp and epoch from filename
            match = re.match(r'surf_maneuver_model_(\d{8}_\d{4})_checkpoint_epoch_(\d+)\.pth', filename)
            if match:
                timestamp, epoch = match.groups()
                checkpoints.append({
                    'filename': filename,
                    'timestamp': timestamp,
                    'epoch': int(epoch)
                })
    
    # Sort by timestamp and epoch
    checkpoints.sort(key=lambda x: (x['timestamp'], x['epoch']))
    return checkpoints

def save_checkpoint(model, optimizer, epoch, timestamp, elapsed_time, class_distribution, is_final=False):
    """Save a checkpoint with model state, optimizer state, and training info."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'timestamp': timestamp,
        'elapsed_time': elapsed_time,
        'class_distribution': class_distribution
    }
    
    if is_final:
        filename = f"../models/surf_maneuver_model_{timestamp}.pth"
    else:
        filename = f"../models/surf_maneuver_model_{timestamp}_checkpoint_epoch_{epoch+1}.pth"
    
    torch.save(checkpoint, filename)
    return filename

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load a checkpoint and validate model architecture.
    
    Handles both old format (just model state dict) and new format (full training state).
    For old format, extracts timestamp and epoch from filename.
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Check if this is an old format checkpoint (just model state dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format - full training state
        model_state = checkpoint['model_state_dict']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        timestamp = checkpoint['timestamp']
        elapsed_time = checkpoint.get('elapsed_time', 0.0)  # Default to 0 for older checkpoints
        class_distribution = checkpoint.get('class_distribution', None)  # None for older checkpoints
    else:
        # Old format - just model state dict
        print("Note: Loading old format checkpoint (no training state).")
        print("Please ensure you're using the same training configuration as the original training.")
        model_state = checkpoint
        
        # Extract timestamp and epoch from filename
        match = re.match(r'surf_maneuver_model_(\d{8}_\d{4})_checkpoint_epoch_(\d+)\.pth', os.path.basename(checkpoint_path))
        if match:
            timestamp, epoch = match.groups()
            epoch = int(epoch)  # Convert to integer
        else:
            raise ValueError("Could not extract timestamp and epoch from checkpoint filename")
        elapsed_time = 0.0  # No time tracking in old format
        class_distribution = None  # No class distribution in old format
    
    # Validate model architecture by comparing state dict keys
    current_model_state = model.state_dict()
    if set(current_model_state.keys()) != set(model_state.keys()):
        raise ValueError("Model architecture in checkpoint does not match current model")
    
    # Load state
    model.load_state_dict(model_state)
    
    return epoch, timestamp, elapsed_time, class_distribution

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Train a surf maneuver detection model.')
parser.add_argument('--mode', choices=['prod', 'dev'], default='dev', help='Set the application mode (prod or dev).')
parser.add_argument('--focal_loss', action='store_true', help='Use Focal Loss instead of weighted Cross Entropy.')
parser.add_argument('--weight_method', choices=['inverse', 'effective', 'sqrt', 'manual', 'balanced', 'none'], default='none', 
                   help='Method for calculating class weights. Use "none" for no weighting.')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma parameter for Focal Loss (if used).')
parser.add_argument('--freeze_backbone', action='store_true', default=True, 
                   help='Freeze the backbone of the model and only train the classifier head. Default is True.')
parser.add_argument('--unfreeze_backbone', action='store_false', dest='freeze_backbone',
                   help='Unfreeze the backbone of the model to train all parameters.')
args = parser.parse_args()
mode = args.mode
use_focal_loss = args.focal_loss
weight_method = args.weight_method
focal_gamma = args.gamma
freeze_backbone = args.freeze_backbone

# Ask user if they want to resume from checkpoint
print("\nDo you want to:")
print("1. Train a new model from scratch")
print("2. Resume training from a checkpoint")
while True:
    try:
        choice = int(input("\nEnter your choice (1 or 2): "))
        if choice in [1, 2]:
            break
        print("Please enter 1 or 2")
    except ValueError:
        print("Please enter a valid number")

# Initialize variables
start_epoch = 0
timestamp = None
total_elapsed_time = 0.0

if choice == 2:
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

# Hyperparameters, defaults for dev mode
batch_size = 1
learning_rate = 0.01
num_epochs = 1
if mode == 'prod':
    batch_size = 8
    learning_rate = 0.005
    num_epochs = 20
print('>  Setting hyperparameters... (mode is: ' + mode + ')')

# Set device to GPU if available, otherwise use CPU
print('>  Configuring device...')
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('>  Using MPS (Apple Silicon GPU) acceleration')
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print('>  Using CUDA (NVIDIA GPU) acceleration')
else:
    device = torch.device("cpu")
    print('>  Using CPU for computation')

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
if weight_method == 'none':
    # No weighting
    use_weights = False
    class_weights = None
    print('>  Using unweighted loss (no class weighting)')
else:
    use_weights = True
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

# Create loss function with class weights
if use_focal_loss:
    print(f'>  Using Focal Loss with gamma={focal_gamma} and class weights')
    criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
elif use_weights:
    print('>  Using Cross Entropy Loss with class weights')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    print('>  Using unweighted Cross Entropy Loss')
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load checkpoint if resuming
if choice == 2:
    checkpoint_path = os.path.join("../models", selected_cp['filename'])
    print(f"\nLoading checkpoint: {checkpoint_path}")
    try:
        start_epoch, timestamp, total_elapsed_time, saved_class_distribution = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Previous training time: {total_elapsed_time:.2f} seconds")
        
        # Use saved class distribution if available
        if saved_class_distribution is not None:
            print("Using class distribution from checkpoint")
            class_distribution = saved_class_distribution
            total_samples = sum(class_distribution.values())
            num_classes = max(class_distribution.keys()) + 1
            
            # Recalculate class weights based on saved distribution
            class_counts = torch.zeros(num_classes)
            for class_id in range(num_classes):
                class_counts[class_id] = class_distribution.get(class_id, 1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training.")
        choice = 1

# Generate timestamp if starting fresh
if choice == 1:
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    timestamp = now.strftime("%Y%m%d_%H%M")

# Add learning rate scheduler for better convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1
)

# Lists to track metrics
epoch_losses = []
batch_losses = []

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
        batch_losses.append(loss_value)
        
        # Print loss every N batches for progress feedback
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"    >  Batch {batch_idx + 1}/{total_batches}, Loss: {loss_value:.4f}")

    # Calculate and track epoch loss
    epoch_loss = running_loss / total_batches
    epoch_losses.append(epoch_loss)
    
    # Update learning rate based on loss
    scheduler.step(epoch_loss)
    
    # Calculate time for this epoch and update total
    epoch_duration = time.time() - epoch_start_time
    total_elapsed_time += epoch_duration
    
    # Print average loss and time taken for the epoch
    print(f"    >  Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds. Average Loss: {epoch_loss:.4f}")    
    print(f"    >  Current learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"    >  Total training time so far: {total_elapsed_time:.2f} seconds")

    # Save model checkpoint for each epoch except the last one
    if epoch < num_epochs - 1:  # Skip the last epoch as it will be saved as the final model
        checkpoint_path = save_checkpoint(model, optimizer, epoch, timestamp, total_elapsed_time, class_distribution)
        print(f"    >  Model checkpoint saved: {checkpoint_path}")
    
    # Reset epoch timer for next epoch
    epoch_start_time = time.time()

print("Training complete.")

# Save final model
model_filename = save_checkpoint(model, optimizer, num_epochs - 1, timestamp, total_elapsed_time, class_distribution, is_final=True)
print(f"Final model saved: {model_filename}")
print(f"Total training time: {total_elapsed_time:.2f} seconds")

# Write training log
log_filename = f"../logs/training_{timestamp}.log"
with open(log_filename, 'w') as f:
    f.write(f"Training Log: surf_maneuver_model_{timestamp}.pth\n")
    f.write("=" * (len(timestamp) + 35) + "\n\n")
    
    # Note about old format checkpoint if applicable
    if choice == 2 and not isinstance(torch.load(os.path.join("../models", selected_cp['filename'])), dict):
        f.write("Note: Resumed from old format checkpoint (no training state saved).\n")
        f.write("Training time and loss history only includes the resumed portion of training.\n\n")
    
    # Configuration section
    f.write("Configuration\n")
    f.write("------------\n")
    f.write(f"Mode: {mode}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Number of epochs: {num_epochs}\n")
    f.write(f"Loss function: {'Focal Loss' if use_focal_loss else 'Cross Entropy Loss'}\n")
    f.write(f"Class weighting: {weight_method}\n")
    if use_focal_loss:
        f.write(f"Focal loss gamma: {focal_gamma}\n")
    f.write(f"Backbone frozen: {freeze_backbone}\n\n")
    
    # Class distribution section
    f.write("Class Distribution\n")
    f.write("-----------------\n")
    for class_id in range(num_classes):
        count = class_distribution.get(class_id, 0)
        percentage = (count / total_samples) * 100
        f.write(f"Class {class_id}: {count} samples ({percentage:.2f}%)\n")
    f.write("\n")
    
    # Training progress section
    f.write("Training Progress\n")
    f.write("----------------\n")
    f.write(f"Total training time: {total_elapsed_time:.2f} seconds\n\n")
    
    f.write("Epoch Losses:\n")
    for i, loss in enumerate(epoch_losses, 1):
        f.write(f"{i}: {loss:.4f}\n")
    f.write("\n")
    
    # Final results section
    f.write("Final Results\n")
    f.write("------------\n")
    f.write(f"Final loss: {epoch_losses[-1]:.4f}\n")
    f.write(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")

print(f"Training log saved to: {log_filename}")
