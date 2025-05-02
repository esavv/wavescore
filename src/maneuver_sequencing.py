# Assumes execution from the /src directory.
# Usage:
# src $ python maneuver_sequencing.py 1Zj_jAPToxI
#
# Some things to do next here:
#  - Add error checking
import os, sys
import pandas as pd
from utils import sequence_video_frames, label_sequences_from_csv

# assert video ID command-line argument is provided
if len(sys.argv) < 2:
    print("Error: Video ID not provided.")
    sys.exit()
vid_id = sys.argv[1]

# assert that the /data/heats/1Zj_jAPToxI directory exists
heat_path = os.path.join('..', 'data', 'heats', vid_id)
if not os.path.exists(heat_path):
    print('Heat directory doesn\'t exist: ' + vid_id)
    sys.exit()
# assert that the /rides/ directory exists
rides_path = os.path.join(heat_path, 'rides')
if not os.path.exists(rides_path):
    print('Rides directory doesn\'t exist for heat ' + vid_id)
    sys.exit()

rides = [d for d in os.listdir(rides_path) if os.path.isdir(os.path.join(rides_path, d))]
rides.sort()

# iterate through each ride
for ride in rides:
    # extract the ride number directly (directory name is now just the number)
    ride_no = ride  

    # check if the video and human labels files are present
    ride_path = os.path.join(rides_path, ride)
    mp4_file = os.path.join(ride_path, vid_id + "_" + ride_no + ".mp4")
    csv_file = os.path.join(ride_path, vid_id + "_" + ride_no + "_human_labels.csv")
    mp4_exists = os.path.exists(mp4_file)
    csv_exists = os.path.exists(csv_file)

    if not mp4_exists and not csv_exists:
        print(ride + ": Skipping because both mp4 and csv files are missing")
    elif not mp4_exists:
        print(ride + ": Skipping because the mp4 file is missing")
    elif not csv_exists:
        print(ride + ": Skipping because the csv file is missing")
    # if all is good, perform maneuver sequencing for this ride
    else:
        print(ride + ": Performing maneuver sequencing!")

        # Create the destination directory
        seqs_path = os.path.join(ride_path, 'seqs')
        
        # Extract frames from video into sequence directories
        sequences_metadata = sequence_video_frames(
            video_path=mp4_file,
            output_dir=seqs_path,
            sequence_duration=2
        )
        
        # Read the human labels CSV
        labels_df = pd.read_csv(csv_file)
        
        # Create and save sequence labels
        labels_file = vid_id + "_" + ride_no + '_seq_labels.csv'
        labels_path = os.path.join(ride_path, labels_file)
        
        # Create labeled sequences
        label_sequences_from_csv(
            sequences_metadata=sequences_metadata,
            labels_df=labels_df,
            output_csv_path=labels_path
        )