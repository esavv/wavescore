# Naively assumes execution from the /data/heats/heat_1Zj_jAPToxI/rides/ride_0 directory
# Some things to do next here:
#  - Instead of doing this for ride_0, loop through all of the rides in /rides/
#  - Add error checking
#  - Remove hardcoding & allow for command-line arguments
import cv2
import math
import os
import pandas as pd
vid_name = '1Zj_jAPToxI_0'
vid_file = vid_name + '.mp4'
human_file = vid_name + '_human_labels.csv'

# helper function to convert 'MM:SS' to seconds
def timestamp_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

# open the video file
cap = cv2.VideoCapture(vid_file)

# get the fps, frame count, video duration, & number of sequences needed
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
sequence_duration = 2
frames_per_sequence = int(sequence_duration * fps)
video_duration = total_frames / fps
total_sequences = math.ceil(video_duration / sequence_duration)

# read the human labels & convert the timestamps
labels_df = pd.read_csv(human_file)
labels_df['timestamp'] = labels_df['timestamp'].apply(timestamp_to_seconds)

# create the destination directory
os.makedirs('seqs', exist_ok=True)

# iterate through each sequence to extract frames
seq_labels = []
frames_remaining = total_frames
for sq in range(total_sequences):
    # create the sequence directory
    seq_dir = f'seq_{sq}'
    os.makedirs('seqs/' + seq_dir, exist_ok=True)

    # infer the start & end timestamps of this sequence:
    start_time = sq * sequence_duration
    end_time = min((sq+1) * sequence_duration, video_duration)

    # determine if this sequence contains a maneuver based on timestamps
    maneuvers_in_seq = labels_df[(labels_df['timestamp'] >= start_time) & (labels_df['timestamp'] < end_time)]
    if len(maneuvers_in_seq) > 0:
        maneuver_type = maneuvers_in_seq.iloc[0]['maneuver_id']  # Use first maneuver in case of overlap
    else:
        maneuver_type = 'Z'
    
    seq_labels.append([f"seq_{sq}", maneuver_type])

    # extract each frame in the sequence & save it
    frame_count = 0
    while frames_remaining > 0 and frame_count < frames_per_sequence:
        ret, frame = cap.read()    
        frame_filename = f"frame_{frame_count:02}.jpg"
        frame_path = 'seqs/' + seq_dir + '/' + frame_filename
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        frames_remaining -= 1

# save the sequence labels
labels_file = vid_name + '_seq_labels.csv'
seq_labels_df = pd.DataFrame(seq_labels, columns=['sequence_id', 'label'])
seq_labels_df.to_csv(labels_file, index=False)
 
cap.release()