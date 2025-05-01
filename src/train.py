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
import argparse, pytz, time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SurfManeuverDataset
from model import SurfManeuverModel

# Set up command-line arguments & configure 'prod' and 'dev' modes (via an environment variable).
parser = argparse.ArgumentParser(description='Toggle between prod and dev modes.')
parser.add_argument('--mode', choices=['prod', 'dev'], default='dev', help='Set the application mode (prod or dev).')
args = parser.parse_args()
mode = args.mode

# Hyperparameters, defaults for dev mode
batch_size = 1
learning_rate = 0.01
num_epochs = 1
if mode == 'prod':
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 10
print('>  Setting hyperparameters... (mode is: ' + mode + ')')

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
print('>  Setting tranform...')
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to match CNN input size
    transforms.ToTensor(),
])

print('>  Creating dataset...')
dataset = SurfManeuverDataset(root_dir="../data/heats", transform=transform, mode=mode)
print('>  Creating dataloader...')
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
model = SurfManeuverModel(num_classes=10, mode=mode)  # Adjust num_classes as needed
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print('>  Starting training...')
for epoch in range(num_epochs):
    print('  >  Epoch ' + str(epoch))
    running_loss = 0.0
    for batch_idx, (frames, labels) in enumerate(dataloader):
        # Print batch progress
        print(f"    >  Processing batch {batch_idx + 1}/{total_batches}")

        # Move data to the correct device
        # print(f"      >  Moving data to the correct device")
        frames, labels = frames.to(device), labels.to(device)
        
        # Forward pass
        # print(f"      >  Forward pass outputs")
        outputs = model(frames)
        # print(f"      >  Forward pass criterion")
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        # print(f"      >  Backward pass zero grad")
        optimizer.zero_grad()
        # print(f"      >  Backward pass backward")
        loss.backward()
        # print(f"      >  Backward pass optimizer step")
        optimizer.step()
        
        # print(f"      >  Computing running loss")
        running_loss += loss.item()
        # Print loss every N batches for progress feedback
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"    >  Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item():.4f}")

    # Print average loss and time taken for the epoch
    epoch_duration = time.time() - start_time
    print(f"    >  Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds. Average Loss: {running_loss / total_batches:.4f}")    

print("Training complete.")

est = pytz.timezone('US/Eastern')
now = datetime.now(est)
timestamp = now.strftime("%Y%m%d_%H%M")

torch.save(model.state_dict(), "../models/surf_maneuver_model_" + timestamp + ".pth")
