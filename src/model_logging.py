"""Training log management for surf maneuver detection model.

This module handles writing training logs with consistent formatting.
"""

import math
from utils import format_time

def write_training_log(log_filename, timestamp, mode, batch_size, learning_rate, num_epochs,
                      use_focal_loss, weight_method, focal_gamma, freeze_backbone,
                      class_distribution, maneuver_names, total_elapsed_time,
                      epoch_losses, epoch_times, final_lr, is_old_format=False):
    """Write a comprehensive training log file.
    
    Args:
        log_filename: Path to write the log file
        timestamp: Training session timestamp
        mode: Training mode ('dev' or 'prod')
        batch_size: Batch size used
        learning_rate: Initial learning rate
        num_epochs: Number of epochs trained
        use_focal_loss: Whether Focal Loss was used
        weight_method: Method used for class weighting
        focal_gamma: Gamma parameter for Focal Loss (if used)
        freeze_backbone: Whether model backbone was frozen
        class_distribution: Distribution of classes in training data
        maneuver_names: Mapping of class IDs to maneuver names
        total_elapsed_time: Total training time in seconds
        epoch_losses: List of loss values per epoch
        epoch_times: List of time taken per epoch
        final_lr: Final learning rate
        is_old_format: Whether training resumed from old format checkpoint
    """
    # Calculate derived values
    total_samples = sum(class_distribution.values())
    max_count_width = max(len(str(count)) for count in class_distribution.values()) + 1
    num_classes = max(class_distribution.keys()) + 1
    
    with open(log_filename, 'w') as f:
        # Write header
        f.write(f"Training Log: surf_maneuver_model_{timestamp}.pth\n")
        f.write("=" * (len(timestamp) + 35) + "\n\n")
        
        # Note about old format checkpoint if applicable
        if is_old_format:
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
            name = maneuver_names.get(class_id, f"Unknown-{class_id}")
            f.write(f"Class {class_id} - {name}: {count:>{max_count_width}} samples ({percentage:>5.2f}%)\n")
        f.write("\n")
        
        # Training progress section
        f.write("Training Progress\n")
        f.write("----------------\n")
        f.write(f"Total training time: {format_time(total_elapsed_time)}\n\n")
        
        # Epoch results in tabular format
        f.write("Epoch Results\n")
        f.write("-------------\n")
        f.write(f"{'Epoch':>6} {'Loss':>10} {'Time':>12} {'Cumulative':>12}\n")
        f.write("-" * 42 + "\n")
        
        cumulative_time = 0
        for i, (loss, epoch_time) in enumerate(zip(epoch_losses, epoch_times), 1):
            cumulative_time += epoch_time
            f.write(f"{i:>6} {loss:>10.4f} {format_time(epoch_time):>12} {format_time(cumulative_time):>12}\n")
        f.write("\n")
        
        # Final results section
        f.write("Final Results\n")
        f.write("------------\n")
        f.write(f"Final loss: {epoch_losses[-1]:.4f}\n")
        f.write(f"Final learning rate: {final_lr:.6f}\n")
        
        # Add random guessing baseline
        random_guess_loss = -math.log(1.0 / num_classes)
        f.write(f"Random guessing baseline: {random_guess_loss:.4f}\n")
        f.write(f"Improvement over random: {((random_guess_loss - epoch_losses[-1]) / random_guess_loss * 100):.1f}%\n\n")
        
        # Add Inference Notes section
        f.write("Inference Notes\n")
        f.write("-------------\n")
        f.write("\n")  # Add blank line for notes 