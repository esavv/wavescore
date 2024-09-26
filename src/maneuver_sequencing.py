# This program converts a video of a surf ride + a set of human-made labels
# identifying maneuvers in the ride into data ready to be fed to a
# ML model for training. In particular, the program converts the video into a set
# of frame sequences and assigns a maneuver label to each sequence, including
# a "no maneuver" label for sequences that, well, don't have a maneuver.
import argparse
import cv2
import os
import pandas as pd
import sys

def check_files(video_dir, ride_dir, video_path, labels_path):
    # Check if the video and ride directories exist
    if not os.path.isdir(video_dir):
        print(f"Error: Video directory not found at {video_dir}")
        sys.exit(1)
    if not os.path.isdir(ride_dir):
        print(f"Error: Ride directory not found at {ride_dir}")
        sys.exit(1)
    
    # Check if the video and labels files exist
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
    if not os.path.isfile(labels_path):
        print(f"Error: Labels CSV file not found at {labels_path}")
        sys.exit(1)

def create_sequence_directories(output_dir, num_sequences):
    seq_dir = os.path.join(output_dir, 'seqs')
    os.makedirs(seq_dir, exist_ok=True)
    for i in range(num_sequences):
        os.makedirs(os.path.join(seq_dir, f"seq{i:02}"), exist_ok=True)
    return seq_dir

def extract_frames(video_path, start_time, end_time, num_frames):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int((end_time - start_time) * fps)
    
    # Calculate frame step to extract `num_frames`
    frame_step = max(1, total_frames // num_frames)
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # Set starting position in milliseconds
    
    while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_step)
        
        if len(frames) >= num_frames:
            break

    cap.release()
    return frames

def save_frames(sequence_dir, frames):
    for i, frame in enumerate(frames):
        frame_path = os.path.join(sequence_dir, f"frame_{i:02}.jpg")
        cv2.imwrite(frame_path, frame)

# Helper function to convert timestamp from 'MM:SS' to seconds
def timestamp_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

def process_video(video_id, ride_id, base_dir="data/heats", sequence_duration=2, num_frames=10):
    # Define the video and ride directories
    video_dir = os.path.join(base_dir, f"heat_{video_id}")
    ride_dir = os.path.join(video_dir, f"rides/ride_{ride_id}")
    
    # Define the video and labels file paths
    video_file = f"{video_id}_{ride_id}.mp4"
    label_file = f"{video_id}_{ride_id}_human_labels.csv"
    video_path = os.path.join(ride_dir, video_file)
    labels_path = os.path.join(ride_dir, label_file)
    
    # Validate directories and files
    check_files(video_dir, ride_dir, video_path, labels_path)
    
    # Read the video and human labels
    cap = cv2.VideoCapture(video_path)
    video_length = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_sequences = int(video_length // sequence_duration) + 1
    cap.release()
    
    # Create sequence directories
    seq_dir = create_sequence_directories(ride_dir, total_sequences)
    
    # Read human labels and convert timestamps to seconds
    labels_df = pd.read_csv(labels_path)
    labels_df['timestamp'] = labels_df['timestamp'].apply(timestamp_to_seconds)
    
    # Initialize labels list for the sequences
    seq_labels = []
    
    # Process each sequence
    for seq_num in range(total_sequences):
        start_time = (seq_num) * sequence_duration
        end_time = min(seq_num+1 * sequence_duration, video_length)
        
        sequence_dir = os.path.join(seq_dir, f"seq{seq_num:02}")
        frames = extract_frames(video_path, start_time, end_time, num_frames)
        save_frames(sequence_dir, frames)
        
        # Determine if this sequence contains a maneuver based on timestamps
        maneuvers_in_seq = labels_df[(labels_df['timestamp'] >= start_time) & (labels_df['timestamp'] < end_time)]
        if len(maneuvers_in_seq) > 0:
            maneuver_type = maneuvers_in_seq.iloc[0]['maneuver']  # Use first maneuver in case of overlap
        else:
            maneuver_type = 'Z'
        
        seq_labels.append([f"seq{seq_num:02}", maneuver_type])
    
    # Save the sequence labels to CSV
    labels_output_path = os.path.join(seq_dir, f'{video_id}_{ride_id}_seq_labels.csv')
    seq_labels_df = pd.DataFrame(seq_labels, columns=['sequence_id', 'manuever_id'])
    seq_labels_df.to_csv(labels_output_path, index=False)
    
    print(f"Processing complete. Sequences and labels saved to {seq_dir}.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process surf ride video into labeled sequences.")
    parser.add_argument("video_id", type=str, help="ID of the video (heat number, as a string)")
    parser.add_argument("ride_id", type=int, help="ID of the ride")
    
    args = parser.parse_args()
    
    # Process video and create labeled frame sequences
    process_video(args.video_id, args.ride_id)
