# This script downloads a YouTube video and sets up the required directory structure
# for processing surf rides. It expects a YouTube video ID as input and optionally
# allows downloading only a specific time range of the video.

# Suppose we're downloading video 123. This script:
#  > Creates directory: data/heats/heat_123
#  > Downloads video: data/heats/heat_123/123.mp4
#  > Creates file: data/heats/heat_123/ride_times_123.csv (with headers only)
#  > Creates file: data/heats/heat_123/rides/ride_0/123_0_human_labels.csv (with headers only)
#  > etc. for each ride directory

# Usage:
# src $ python download_youtube.py 123

import os, subprocess, sys, csv

def create_directory_structure(video_id):
    """Create the required directory structure and CSV file."""
    # Create the heat directory
    heat_dir = os.path.join('..', 'data', 'heats', f'heat_{video_id}')
    os.makedirs(heat_dir, exist_ok=True)
    
    # Create the CSV file with headers
    csv_path = os.path.join(heat_dir, f'ride_times_{video_id}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start', 'end'])
    
    return heat_dir

def download_video(video_id, heat_dir, start_time=None, end_time=None):
    """Download the video using yt-dlp."""
    output_path = os.path.join(heat_dir, f'{video_id}.mp4')
    
    # Base command
    command = [
        'yt-dlp',
        '-f', 'mp4',
        '-o', output_path,
    ]
    
    # Add time range if specified
    if start_time and end_time:
        command.extend(['--download-sections', f'*{start_time}-{end_time}'])
    
    # Add the video URL
    command.append(f'https://www.youtube.com/watch?v={video_id}')
    
    # Run the command
    print(f"Downloading video {video_id}...")
    subprocess.run(command, check=True)
    print(f"Video downloaded to {output_path}")

def main():
    # Check if video ID is provided
    if len(sys.argv) < 2:
        print("Error: YouTube video ID not provided.")
        print("Usage: python download_youtube.py <video_id>")
        sys.exit(1)
    
    video_id = sys.argv[1]
    
    # Ask user if they want the full video or a subset
    print("\nDownload options:")
    print("1. Full video")
    print("2. Specific time range")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        # Create directory structure and download full video
        heat_dir = create_directory_structure(video_id)
        download_video(video_id, heat_dir)
    elif choice == '2':
        # Get time range from user
        start_time = input("Enter start time (HH:MM:SS): ")
        end_time   = input("Enter end time (HH:MM:SS):   ")
        
        # Create directory structure and download video subset
        heat_dir = create_directory_structure(video_id)
        download_video(video_id, heat_dir, start_time, end_time)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    print("\nDirectory structure created:")
    print(f"- Heat directory: {heat_dir}")
    print(f"- Video file: {video_id}.mp4")
    print(f"- CSV file: ride_times_{video_id}.csv")

if __name__ == "__main__":
    main() 