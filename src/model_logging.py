"""Training log management for surf maneuver detection model.

This module handles writing training logs with consistent formatting.
"""

import math
from utils import format_time

def write_training_log(log_filename, timestamp, mode, batch_size, learning_rate, num_epochs,
                      total_elapsed_time, epoch_losses, epoch_times,
                      model_type='maneuver', variant='base', loss_function='mse',
                      freeze_backbone=True, use_focal_loss=False, weight_method=None,
                      focal_gamma=None, class_distribution=None, maneuver_names=None,
                      final_lr=None, is_old_format=False):
    """Write a comprehensive training log file.
    
    Args:
        log_filename: Path to write the log file
        timestamp: Training session timestamp
        mode: Training mode ('dev' or 'prod')
        batch_size: Batch size used
        learning_rate: Initial learning rate
        num_epochs: Number of epochs trained
        total_elapsed_time: Total training time in seconds
        epoch_losses: List of loss values per epoch
        epoch_times: List of time taken per epoch
        model_type: Type of model ('maneuver' or 'score')
        variant: Model variant ('base' or 'large')
        loss_function: Loss function used ('mse', 'mae', 'huber', or 'cross_entropy')
        freeze_backbone: Whether model backbone was frozen
        use_focal_loss: Whether Focal Loss was used (for maneuver prediction)
        weight_method: Method used for class weighting (for maneuver prediction)
        focal_gamma: Gamma parameter for Focal Loss (if used)
        class_distribution: Distribution of classes in training data (for maneuver prediction)
        maneuver_names: Mapping of class IDs to maneuver names (for maneuver prediction)
        final_lr: Final learning rate
        is_old_format: Whether training resumed from old format checkpoint
    """
    with open(log_filename, 'w') as f:
        # Write header
        model_name = "maneuver_model" if model_type == "maneuver" else "score_model"
        f.write(f"Training Log: {model_name}_{timestamp}.pth\n")
        f.write("=" * (len(timestamp) + 35) + "\n\n")
        
        # Note about old format checkpoint if applicable
        if is_old_format:
            f.write("Note: Resumed from old format checkpoint (no training state saved).\n")
            f.write("Training time and loss history only includes the resumed portion of training.\n\n")
        
        # Configuration section
        f.write("Configuration\n")
        f.write("------------\n")
        f.write(f"Mode: {mode}\n")
        if model_type == "score":
            f.write(f"Model: {variant.upper()}-{variant}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        
        # Loss function section
        if model_type == "maneuver":
            f.write(f"Loss function: {'Focal Loss' if use_focal_loss else 'Cross Entropy Loss'}\n")
            f.write(f"Class weighting: {weight_method}\n")
            if use_focal_loss:
                f.write(f"Focal loss gamma: {focal_gamma}\n")
        else:
            f.write(f"Loss function: {loss_function.upper()}\n")
        
        f.write(f"Backbone frozen: {freeze_backbone}\n\n")
        
        # Class distribution section (only for maneuver prediction)
        if model_type == "maneuver" and class_distribution and maneuver_names:
            f.write("Class Distribution\n")
            f.write("-----------------\n")
            total_samples = sum(class_distribution.values())
            max_count_width = max(len(str(count)) for count in class_distribution.values()) + 1
            num_classes = max(class_distribution.keys()) + 1
            
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
        if final_lr:
            f.write(f"Final learning rate: {final_lr:.6f}\n")
        
        # Add random guessing baseline
        if model_type == "maneuver":
            random_guess_loss = -math.log(1.0 / num_classes)
            f.write(f"Random guessing baseline: {random_guess_loss:.4f}\n")
        else:
            random_guess_loss = 5.0  # MSE of predicting mean score (0-10)
            f.write(f"Random guessing baseline (MSE): {random_guess_loss:.4f}\n")
        
        f.write(f"Improvement over random: {((random_guess_loss - epoch_losses[-1]) / random_guess_loss * 100):.1f}%\n\n")
        
        # Add Inference Notes section
        f.write("Inference Notes\n")
        f.write("-------------\n")
        f.write("\n")  # Add blank line for notes

def write_score_training_log(log_filename, timestamp, mode, model_type, variant, batch_size, learning_rate, num_epochs,
                           loss_function, freeze_backbone, total_elapsed_time, epoch_losses, epoch_times):
    """Write a comprehensive training log file for score prediction.
    
    Args:
        log_filename: Path to write the log file
        timestamp: Training session timestamp
        mode: Training mode ('dev' or 'prod')
        model_type: Type of model ('clip' or 'vit')
        variant: Model variant ('base' or 'large')
        batch_size: Batch size used
        learning_rate: Initial learning rate
        num_epochs: Number of epochs trained
        loss_function: Loss function used ('mse', 'mae', or 'huber')
        freeze_backbone: Whether model backbone was frozen
        total_elapsed_time: Total training time in seconds
        epoch_losses: List of loss values per epoch
        epoch_times: List of time taken per epoch
    """
    with open(log_filename, 'w') as f:
        # Write header
        f.write(f"Training Log: score_model_{timestamp}.pth\n")
        f.write("=" * (len(timestamp) + 35) + "\n\n")
        
        # Configuration section
        f.write("Configuration\n")
        f.write("------------\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Model: {model_type.upper()}-{variant}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Loss function: {loss_function.upper()}\n")
        f.write(f"Backbone frozen: {freeze_backbone}\n\n")
        
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
        
        # Add random guessing baseline (MSE of predicting mean score)
        random_guess_loss = 5.0  # Assuming scores are 0-10, mean is 5.0
        f.write(f"Random guessing baseline (MSE): {random_guess_loss:.4f}\n")
        f.write(f"Improvement over random: {((random_guess_loss - epoch_losses[-1]) / random_guess_loss * 100):.1f}%\n\n")
        
        # Add Inference Notes section
        f.write("Inference Notes\n")
        f.write("-------------\n")
        f.write("\n")  # Add blank line for notes 