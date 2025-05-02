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

        # Print the root directory for debugging
        print(f"  $    Loading data from root directory: {self.root_dir}")

        # Iterate through each surfing heat
        for heat_dir in os.listdir(root_dir):
            heat_path = os.path.join(root_dir, heat_dir)
            if not os.path.isdir(heat_path):
                continue

            heat_id = heat_dir  # heat_id is now the same as the directory name

            rides_dir = os.path.join(heat_path, "rides")
            # Iterate through each ride in the heat
            for ride_dir in os.listdir(rides_dir):
                ride_path = os.path.join(rides_dir, ride_dir)
                ride_id = ride_dir  # ride_id is now directly the directory name
                seq_labels_path = os.path.join(ride_path, "seq_labels.csv")
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
