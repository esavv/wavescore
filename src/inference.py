import torch
from torchvision import transforms
from dataset import SurfManeuverDataset
from model import SurfManeuverModel
from PIL import Image
import os

# Define transformations (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the saved model
model_path = "models/test_surf_maneuver_model.pth"
model = SurfManeuverModel(num_classes=10)  # Ensure num_classes matches your trained model
model.load_state_dict(torch.load(model_path))
model.eval()

def load_sequence(seq_dir):
    """Load frames from a sequence directory and apply transforms."""
    frames = []
    for frame_file in sorted(os.listdir(seq_dir)):
        frame_path = os.path.join(seq_dir, frame_file)
        image = Image.open(frame_path).convert("RGB")
        image = transform(image)
        frames.append(image)
    return torch.stack(frames).unsqueeze(0)  # Shape (1, num_frames, channels, height, width)

def infer_sequence(model, seq_dir):
    """Run inference on a single sequence."""
    sequence = load_sequence(seq_dir)
    with torch.no_grad():
        output = model(sequence)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

# Define path to new video sequences
new_video_seqs = "path/to/new/video/seqs"  # Replace with your sequence directory

# Run inference on each sequence in the new video
for seq_name in sorted(os.listdir(new_video_seqs)):
    seq_dir = os.path.join(new_video_seqs, seq_name)
    if os.path.isdir(seq_dir):
        prediction = infer_sequence(model, seq_dir)
        print(f"Predicted maneuver for {seq_name}: {prediction}")
