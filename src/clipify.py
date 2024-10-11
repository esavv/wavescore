# This program converts videos of full surfing heats into clips of individual
# waves ridden, expecting as input both the heat video and a csv denoting the
# start & end times of each ridden wave 
import argparse
import csv
import os
import subprocess

# Main function
# This function is expected to be executed from the parent /surfjudge/ directory
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process video clips based on start and end timestamps from a CSV file.")
    
    # Add arguments for the input video and CSV file
    #  - The 'input_video' argument should be a file in the /data/heats directory, and should be of the format 'data/heats/heat_vid01/vid01.mp4'
    #  - The 'csv_file' argument should be a file in the /data/heats directory, and should be of the format 'data/heats/heat_vid01/ride_times_vid01.csv'
    parser.add_argument('input_video', type=str, help="The input video file to process")
    parser.add_argument('csv_file', type=str, help="The CSV file containing start and end timestamps")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Process the clips

    # Open the CSV file and read the start and end times
    with open(args.csv_file, mode='r') as file:
        reader = csv.DictReader(file)

        base_name = os.path.basename(args.input_video)
        video_name = os.path.splitext(base_name)[0]

        base_path = "data/heats/heat_" + video_name + "/"
        # Create a new rides directory, assumes it doesn't already exist
        os.mkdir(base_path + "rides")

        # Loop through each row in the CSV
        for index, row in enumerate(reader):
            start_time = row['start']
            end_time = row['end']

            ride_path = base_path + f"rides/ride_{index}"
            # Create a new directory for each individual ride (these will later contain maneuver labels + frame sequences for training)
            os.mkdir(ride_path)

            # Define the output filename for each clip
            output_path = ride_path + "/" + video_name + f"_{index}.mp4"

            # Construct the ffmpeg command
            command = [
                "ffmpeg",
                "-i", args.input_video,
                "-ss", start_time,
                "-to", end_time,
                "-c:v", "libx264",
                "-c:a", "aac",
                output_path
            ]

            # Print the command (for debugging)
            print(f"Processing clip {index}: {start_time} to {end_time} -> {output_path}")
    
            # Run the ffmpeg command
            subprocess.run(command, check=True)

    print("All clips processed.")

# Entry point of the script
if __name__ == "__main__":
    main()