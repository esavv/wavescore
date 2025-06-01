# This script trains a deep learning model to predict surf competition scores from video footage.
# The model takes a video of a surf ride as input and outputs a predicted score between 0.0 and 10.0.

# Usage:
#     python score_train.py --mode [dev|prod]

import argparse, os, pytz, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from score_model import VideoScorePredictor
from score_dataset import ScoreDataset
from collate import collate_variable_length_videos
from checkpoints import save_checkpoint, load_checkpoint, get_available_checkpoints
from model_logging import write_training_log

# Constants for model choice
TRAIN_FROM_SCRATCH = 1
RESUME_FROM_CHECKPOINT = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Train surf score prediction model')
    parser.add_argument('--mode', type=str, choices=['dev', 'prod'], default='dev',
                      help='dev mode for faster iteration, prod for full training (default: dev)')
    parser.add_argument('--unfreeze_backbone', action='store_true',
                      help='Unfreeze the backbone of the model to train all parameters. Default is False (backbone frozen).')
    parser.add_argument('--loss', type=str, choices=['mse', 'mae', 'huber'], default='mse',
                      help='loss function to use (default: mse)')
    parser.add_argument('--model_type', type=str, choices=['timesformer', 'video_swin'], default='timesformer',
                      help='model type to use (default: timesformer)')
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch data (videos, attention_mask, scores)
        videos, attention_mask, scores = batch_data
        videos = videos.to(device)
        attention_mask = attention_mask.to(device)
        scores = scores.to(device)
        
        # Forward pass with attention mask
        outputs = model(videos, attention_mask=attention_mask)
        
        optimizer.zero_grad()
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def main():
    args = parse_args()
    freeze_backbone = not args.unfreeze_backbone  # Invert the flag to get freeze_backbone
    
    print(f'> Starting score prediction training in {args.mode} mode')
    
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

    # Hyperparameters, defaults for dev mode if not loaded from checkpoint
    if model_choice == TRAIN_FROM_SCRATCH:
        batch_size = 1
        learning_rate = 0.01
        num_epochs = 1
        loss_function = args.loss
        if args.mode == 'prod':
            batch_size = 8
            learning_rate = 0.005
            num_epochs = 10
        
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
            
            # Load training history
            epoch_losses = training_history['epoch_losses']
            epoch_times = training_history['epoch_times']
            total_elapsed_time = training_history['total_elapsed_time']
            
            # Apply saved training config
            print("Using training configuration from checkpoint")
            args.mode = training_config['mode']
            batch_size = training_config['batch_size']
            learning_rate = training_config['learning_rate']
            num_epochs = training_config['num_epochs']
            freeze_backbone = training_config['freeze_backbone']
            loss_function = training_config['loss_function']
            
            print(f"Resuming from epoch {start_epoch}")
            print(f"Previous training time: {total_elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'> Using device: {device}')
    
    # Initialize model
    print('> Initializing model...')
    model = VideoScorePredictor(model_type=args.model_type, variant='base', freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Initialize dataset and dataloader
    print('> Loading dataset...')
    dataset = ScoreDataset(root_dir="../data/heats", transform=None, model_type=args.model_type)
    print('> Creating dataloader...')
    
    # Create dataloader with fixed batch size
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    print(f'>   Using batch_size={batch_size}')
    
    # Initialize optimizer and loss function
    print('> Setting up optimizer and loss function...')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up loss function based on choice
    if loss_function == 'mse':
        criterion = nn.MSELoss()
        loss_name = 'MSE Loss'
    elif loss_function == 'mae':
        criterion = nn.L1Loss()
        loss_name = 'MAE Loss'
    elif loss_function == 'huber':
        criterion = nn.HuberLoss()
        loss_name = 'Huber Loss'
    
    # Load model and optimizer state if resuming
    if model_choice == RESUME_FROM_CHECKPOINT:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Print final configuration
    print('\n>  Training configuration:')
    print('>    Mode:', args.mode)
    print('>    Batch size:', batch_size)
    print('>    Learning rate:', learning_rate)
    print('>    Number of epochs:', num_epochs)
    print('>    Loss function:', loss_name)
    print('>    Backbone frozen:', freeze_backbone)
    print()
    
    # Training loop
    print('> Starting training loop...')
    
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training
        print('> Training...')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'> Training loss: {train_loss:.4f}')
        
        # Update training history
        epoch_losses.append(train_loss)
        epoch_times.append(0.0)  # TODO: Implement proper time tracking
        total_elapsed_time += 0.0  # TODO: Implement proper time tracking
        
        # Log training progress
        write_training_log(
            timestamp=timestamp,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=None,  # We're not doing validation
            epoch_time=0.0,  # TODO: Implement proper time tracking
            total_time=total_elapsed_time,
            model_type='score_prediction',
            mode=args.mode,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save checkpoint after each epoch
        print('> Saving checkpoint...')
        training_config = {
            'mode': args.mode,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'freeze_backbone': freeze_backbone,
            'loss_function': loss_function
        }
        training_history = {
            'epoch_losses': epoch_losses,
            'epoch_times': epoch_times,
            'total_elapsed_time': total_elapsed_time
        }
        checkpoint_path = save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            epoch,
            train_loss,
            'score_model',
            timestamp=timestamp,
            training_config=training_config,
            training_history=training_history
        )
        print(f"    >  Model checkpoint saved: {checkpoint_path}")
    
    print('\n> Training complete!')

if __name__ == '__main__':
    main() 