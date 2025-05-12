"""Checkpoint management for surf maneuver detection model.

This module handles saving, loading, and managing model checkpoints.
It supports both old format (just model state dict) and new format (full training state).
"""

import os, re, torch

def get_available_checkpoints():
    """List available checkpoints in the models directory.
    
    Returns:
        list: List of dictionaries containing checkpoint info:
            - filename: str
            - timestamp: str
            - epoch: int
    """
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
    """Save a checkpoint with model state, optimizer state, and training info.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        timestamp: Training session timestamp
        elapsed_time: Total training time so far
        class_distribution: Distribution of classes in training data
        is_final: Whether this is the final model (True) or a checkpoint (False)
    
    Returns:
        str: Path to the saved checkpoint file
    """
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
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        tuple: (epoch, timestamp, elapsed_time, class_distribution)
    
    Raises:
        ValueError: If model architecture doesn't match checkpoint
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