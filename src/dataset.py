import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SurfManeuverDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Gather sequence directories and labels
        self.samples = []

        # Iterate through each surfing heat
        for heat_dir in os.listdir(root_dir):
            heat_path = os.path.join(root_dir, heat_dir)
            if not os.path.isdir(heat_path):
                continue

            heat_id = heat_dir.split('_')[-1]  # Assuming the heat ID is at the end of the heat directory name

            rides_dir = os.path.join(heat_path, "rides")
            # Iterate through each ride in the heat
            for ride_dir in os.listdir(rides_dir):
                ride_path = os.path.join(rides_dir, ride_dir)
                ride_id = ride_dir.split('_')[-1] # Extract ride_id from the ride directory name
                seq_labels_path = os.path.join(ride_path, f"{heat_id}_{ride_id}_seq_labels.csv")
                seqs_dir = os.path.join(ride_path, "seqs")
                
                if os.path.exists(seq_labels_path):
                    labels_df = pd.read_csv(seq_labels_path)
                    for _, row in labels_df.iterrows():
                        seq_dir = os.path.join(seqs_dir, row["sequence_id"])
                        self.samples.append((seq_dir, row["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        frames = []

        # Load each frame in the sequence directory
        for frame_file in sorted(os.listdir(seq_dir)):
            frame_path = os.path.join(seq_dir, frame_file)
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        # Stack frames into a tensor with shape (num_frames, channels, height, width)
        return frames, label
