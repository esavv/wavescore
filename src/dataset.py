import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class SurfManeuverDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='dev'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Gather sequence directories and labels
        self.samples = []

        # Print the root directory for debugging
        print(f"  $    Loading data from root directory: {self.root_dir}")

        # Iterate through each surfing heat
        for heat_dir in os.listdir(root_dir):
            heat_path = os.path.join(root_dir, heat_dir)
            if not os.path.isdir(heat_path):
                continue

            heat_id = heat_dir[len("heat_"):]  # Assuming the heat ID is everything after "heat_" in <heat_dir>

            rides_dir = os.path.join(heat_path, "rides")
            # Iterate through each ride in the heat
            for ride_dir in os.listdir(rides_dir):
                ride_path = os.path.join(rides_dir, ride_dir)
                ride_id = ride_dir.split('_')[-1] # Extract ride_id from the ride directory name
                seq_labels_path = os.path.join(ride_path, f"{heat_id}_{ride_id}_seq_labels.csv")
                seqs_dir = os.path.join(ride_path, "seqs")
                
                # Debug information for each ride and label path
                print(f"  $      Looking for sequence labels at: {seq_labels_path}")

                if os.path.exists(seq_labels_path):
                    labels_df = pd.read_csv(seq_labels_path)

                    # Check if labels are found in the DataFrame
                    print(f"  $      Found {len(labels_df)} labels in {seq_labels_path}")

                    for _, row in labels_df.iterrows():
                        seq_dir = os.path.join(seqs_dir, row["sequence_id"])

                        # Verify the existence of the sequence directory
                        if os.path.isdir(seq_dir):
                            # print(f"  $        Adding sample: {seq_dir} with label {row['label']}")
                            self.samples.append((seq_dir, row["label"]))
                        else:
                            print(f"  $        Warning: Sequence directory {seq_dir} does not exist.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        frames = load_frames_from_sequence(seq_dir, self.transform, self.mode)
        return frames, label

def load_frames_from_sequence(seq_dir, transform=None, mode='dev', add_batch_dim=False):
    """Load and transform frames from a sequence directory based on mode settings."""
    SKIP_FREQ = 10,  # skip every 10 frames in dev mode
    COLOR = "L",     # "L" mode is for grayscale in dev mode
    if mode == 'prod':
        SKIP_FREQ =  1
        COLOR = "RGB"
    MAX_LENGTH = 60 / SKIP_FREQ  # base max sequence length
    
    # Load each frame in the sequence directory
    frames = []
    for frame_idx, frame_file in enumerate(sorted(os.listdir(seq_dir))):
        # Skip frames based on mode settings
        if frame_idx % SKIP_FREQ != 0:
            continue

        frame_path = os.path.join(seq_dir, frame_file)
        image = Image.open(frame_path).convert(COLOR)
        if transform:
            image = transform(image)
        frames.append(image)

    # Pad or truncate frames
    frames = pad_sequence(frames, MAX_LENGTH)
    
    # Add batch dimension if needed (for inference)
    if add_batch_dim:
        frames = frames.unsqueeze(0)
        
    return frames

def pad_sequence(frames, max_length=60):
    """Pad or truncate a sequence of frames to the specified length."""
    num_frames = len(frames)
    if num_frames < max_length:
        # Pad with zero tensors to reach max_length
        padding = [torch.zeros_like(frames[0]) for _ in range(max_length - num_frames)]
        frames = frames + padding
    else:
        # Truncate to max_length if too long
        frames = frames[:max_length]
    return torch.stack(frames)