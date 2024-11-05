print('>  Importing modules...')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SurfManeuverDataset
from model import SurfManeuverModel

from datetime import datetime
import pytz

# Hyperparameters
print('>  Setting hyperparameters...')
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Data preparation
print('>  Setting tranform...')
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to match CNN input size
    transforms.ToTensor(),
])

print('>  Creating dataset...')
dataset = SurfManeuverDataset(root_dir="../data/heats", transform=transform)
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

# Model, loss function, and optimizer
print('>  Defining the model...')
model = SurfManeuverModel(num_classes=10)  # Adjust num_classes as needed
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print('>  Starting training...')
for epoch in range(num_epochs):
    print('  >  Epoch ' + str(epoch))
    running_loss = 0.0
    for frames, labels in dataloader:
        # Move data to the correct device
        frames, labels = frames.to('cuda'), labels.to('cuda')
        model = model.to('cuda')
        
        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training complete.")

est = pytz.timezone('US/Eastern')
now = datetime.now(est)
timestamp = now.strftime("%Y%m%d_%H%M")

torch.save(model.state_dict(), "../models/surf_maneuver_model_" + timestamp + ".pth")
