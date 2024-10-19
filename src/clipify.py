# This program converts videos of full surfing heats into clips of individual
# waves ridden, expecting as input both the heat video and a csv denoting the
# start & end times of each ridden wave 

# Suppose we're making clips of video 123.mp4. This script expects the following
# things to exist:
#  > Directory: data/heats/heat_123
#  > File:      data/heats/heat_123/123.mp4
#  > File:      data/heats/heat_123/ride_times_123.csv
import csv, os, subprocess, sys

# assert video ID command-line argument is provided
if len(sys.argv) < 2:
    print("Error: Video ID not provided.")
    sys.exit()
vid_id = sys.argv[1]
current_dir = os.getcwd()

# assert that the /data/heats/heat_1Zj_jAPToxI directory exists
heat_path = os.path.join(current_dir, 'data/heats/heat_' + vid_id)
if not os.path.exists(heat_path):
    print('Heat directory doesn\'t exist: ' + 'heat_' + vid_id)
    sys.exit()

# assert that the video & csv files exist
mp4_file = os.path.join(heat_path, vid_id + ".mp4")
csv_file = os.path.join(heat_path, "ride_times_" + vid_id + ".csv")
if not os.path.exists(mp4_file):
    print('Video file doesn\'t exist: ' + vid_id + ".mp4")
    sys.exit()
if not os.path.exists(csv_file):
    print('Clips csv file doesn\'t exist: ' + "ride_times_" + vid_id + ".csv")
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
        ride_path = os.path.join(rides_path, f"ride_{index}")
        os.makedirs(ride_path, exist_ok=True)

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
            clip_path
        ]

        # Print the command (for debugging)
        print(f"Processing clip {index}: {start_time} to {end_time} -> {clip_path}")

        # Run the ffmpeg command
        subprocess.run(command, check=True)

print("All clips processed.")