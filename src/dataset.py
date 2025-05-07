import os
import pandas as pd
from torch.utils.data import Dataset
from utils import load_frames_from_sequence

class SurfManeuverDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='dev'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Gather sequence directories and labels
        self.samples = []

        print(f">  Loading data from root directory: {self.root_dir}")

        # Iterate through each surfing heat
        for heat_dir in os.listdir(root_dir):
            heat_path = os.path.join(root_dir, heat_dir)
            if not os.path.isdir(heat_path):
                continue

            heat_id = heat_dir  # heat_id is now the same as the directory name
            rides_dir = os.path.join(heat_path, "rides")
            
            # Track stats for this heat
            heat_rides = 0
            heat_labels = 0
            
            # Iterate through each ride in the heat
            for ride_dir in os.listdir(rides_dir):
                ride_path = os.path.join(rides_dir, ride_dir)
                ride_id = ride_dir
                seq_labels_path = os.path.join(ride_path, "seq_labels.csv")
                seqs_dir = os.path.join(ride_path, "seqs")
                
                if os.path.exists(seq_labels_path):
                    labels_df = pd.read_csv(seq_labels_path)
                    heat_rides += 1
                    heat_labels += len(labels_df)

                    for _, row in labels_df.iterrows():
                        seq_dir = os.path.join(seqs_dir, row["sequence_id"])
                        
                        # Verify the existence of the sequence directory
                        if os.path.isdir(seq_dir):
                            self.samples.append((seq_dir, row["label"]))
                        else:
                            print(f">  Warning: Sequence directory {seq_dir} does not exist.")
            
            # Print summary for this heat
            print(f">    Heat '{heat_id}': {heat_rides} rides, {heat_labels} sequence labels")

        print(f">  Total dataset: {len(self.samples)} labeled sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        frames = load_frames_from_sequence(seq_dir, self.transform, self.mode)
        return frames, label
