"""Checkpoint management for surf maneuver detection model.

This module handles saving, loading, and managing model checkpoints.
It supports both old format (just model state dict) and new format (full training state).
"""

import os, re, torch

# Determine the device to use
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint_dir = "../../models"

def get_available_checkpoints():
    """List available checkpoints in the models directory.
    
    Returns:
        list: List of dictionaries containing checkpoint info:
            - filename: str
            - timestamp: str
            - epoch: int
    """
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

def save_checkpoint(model, optimizer, epoch, timestamp, class_distribution, training_config, is_final=False, training_history=None):
    """Save a checkpoint with model state, optimizer state, and training info.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        timestamp: Training session timestamp
        class_distribution: Distribution of classes in training data (for maneuver prediction)
        training_config: Dict containing training hyperparameters
        is_final: Whether this is the final model (True) or a checkpoint (False)
        training_history: Dictionary containing epoch_losses, epoch_times, and total_elapsed_time
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'timestamp': timestamp,
        'training_config': training_config,
        'training_history': training_history
    }
    
    # Get model name from config
    model_name = training_config.get('model_name', 'maneuver_model')
    
    if is_final:
        filename = f"{checkpoint_dir}/{model_name}_{timestamp}.pth"
    else:
        filename = f"{checkpoint_dir}/{model_name}_{timestamp}_checkpoint_epoch_{epoch+1}.pth"
    
    torch.save(checkpoint, filename)
    return filename

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return its contents.
    
    Args:
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        tuple: (model_state, optimizer_state, epoch, timestamp, training_config, training_history)
    """
    # Load checkpoint with appropriate device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is an old format checkpoint (just model state dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format - full training state
        model_state = checkpoint['model_state_dict']
        optimizer_state = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        timestamp = checkpoint['timestamp']
        training_config = checkpoint.get('training_config', None)  # None for older checkpoints
        training_history = checkpoint.get('training_history', None)  # None for older checkpoints
    else:
        # Old format - just model state dict
        print("Note: Loading old format checkpoint (no training state).")
        print("Please ensure you're using the same training configuration as the original training.")
        model_state = checkpoint
        optimizer_state = None
        
        # Extract timestamp and epoch from filename
        match = re.match(r'surf_maneuver_model_(\d{8}_\d{4})_checkpoint_epoch_(\d+)\.pth', os.path.basename(checkpoint_path))
        if match:
            timestamp, epoch = match.groups()
            epoch = int(epoch)  # Convert to integer
        else:
            raise ValueError("Could not extract timestamp and epoch from checkpoint filename")
        training_config = None  # No training config in old format
        training_history = None  # No training history in old format
    
    return model_state, optimizer_state, epoch, timestamp, training_config, training_history 