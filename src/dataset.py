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
        frames = []
        SKIP_FREQ = 10

        # Load each frame in the sequence directory
        for frame_idx, frame_file in enumerate(sorted(os.listdir(seq_dir))):
            # Skip frames in dev mode to speed up training
            if self.mode == 'dev' and frame_idx % SKIP_FREQ != 0:
                continue

            frame_path = os.path.join(seq_dir, frame_file)
            if self.mode == 'prod':
                image = Image.open(frame_path).convert("RGB")
            elif self.mode == 'dev':
                image = Image.open(frame_path).convert("L")  # "L" mode is for grayscale; reduce image size for faster training in dev mode
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # Pad or truncate frames
        max_length = 60
        if self.mode == 'dev':
            max_length = max_length // SKIP_FREQ
        frames = pad_sequence(frames, max_length)

        # Stack frames into a tensor with shape (num_frames, channels, height, width)
        return frames, label
    
def pad_sequence(frames, max_length=60):
    num_frames = len(frames)
    if num_frames < max_length:
        # Pad with zero tensors to reach max_length
        padding = [torch.zeros_like(frames[0]) for _ in range(max_length - num_frames)]
        frames = frames + padding
    else:
        # Truncate to max_length if too long
        frames = frames[:max_length]
    return torch.stack(frames)