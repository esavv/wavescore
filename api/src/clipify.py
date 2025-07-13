# This script converts videos of full surfing heats into clips of individual
# waves ridden, expecting as input both the heat video and a csv denoting the
# start & end times of each ridden wave 

# Suppose we're making clips of video 123.mp4. This script expects the following
# things to exist:
#  > Directory: data/heats/123
#  > File:      data/heats/123/123.mp4
#  > File:      data/heats/123/ride_times.csv

# Usage:
# src $ python clipify.py 123
import csv, os, re, subprocess, sys

data_dir = "../../data"

# assert video ID command-line argument is provided
if len(sys.argv) < 2:
    print("Error: Video ID not provided.")
    sys.exit()
vid_id = sys.argv[1]

# assert that the /data/heats/1Zj_jAPToxI directory exists
heat_path = data_dir + '/heats/' + vid_id
if not os.path.exists(heat_path):
    print('Heat directory doesn\'t exist: ' + vid_id)
    sys.exit()

# assert that the video & csv files exist
mp4_file = os.path.join(heat_path, vid_id + ".mp4")
csv_file = os.path.join(heat_path, "ride_times.csv")
if not os.path.exists(mp4_file):
    print('Video file doesn\'t exist: ' + vid_id + ".mp4")
    sys.exit()
if not os.path.exists(csv_file):
    print('Clips csv file doesn\'t exist: ride_times.csv')
    sys.exit()

# Process the clips
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)

    # Create a new rides directory
    rides_path = os.path.join(heat_path, 'rides')
    os.makedirs(rides_path, exist_ok=True)

    # Loop through each row in the CSV
    for index, row in enumerate(reader):
        start_time = row['start']
        end_time = row['end']

        # Create a new directory for each individual ride (these will later contain maneuver labels + frame sequences for training)
        ride_path = os.path.join(rides_path, f"{index}")
        os.makedirs(ride_path, exist_ok=True)

        # Create the human labels CSV file
        labels_path = os.path.join(ride_path, "human_labels.csv")
        with open(labels_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['start', 'end', 'maneuver_id'])

        # Define the output filename for each clip
        clip_path = os.path.join(ride_path, vid_id + f"_{index}.mp4")

        # Construct the ffmpeg command
        command = [
            "ffmpeg",
            "-i", mp4_file,
            "-ss", start_time,
            "-to", end_time,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-progress", "pipe:1",  # Output progress to stdout
            clip_path
        ]

        # Print the command (for debugging)
        print(f"Processing clip {index}: {start_time} to {end_time} -> {clip_path}")

        # Run the ffmpeg command and capture progress
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Parse progress output
        while True:
            line = process.stdout.readline()
            if not line:
                break
            if "out_time_ms" in line:
                # Extract time in seconds
                time_match = re.search(r"out_time_ms=(\d+)", line)
                if time_match:
                    time_ms = int(time_match.group(1))
                    time_sec = time_ms / 1000000
                    print(f"\rProgress: {time_sec:.1f}s", end="", flush=True)
        
        # Wait for process to complete
        process.wait()
        print()  # New line after progress

print("All clips processed.")