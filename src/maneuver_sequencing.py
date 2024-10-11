# Naively assumes execution from the /data/heats/heat_1Zj_jAPToxI/rides/ride_0 directory
# Some things to do next here:
#  - Convert the human_labels into labels for each sequence
#  - Instead of doing this for ride_0, loop through all of the rides in /rides/
#  - Add error checking
#  - Remove hardcoding & allow for command-line arguments
import cv2
import math
import os
file = '1Zj_jAPToxI_0.mp4'

# open the file
cap = cv2.VideoCapture(file)

# get the fps, frame count, video duration, & number of sequences needed
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
sequence_duration = 2
frames_per_sequence = int(sequence_duration * fps)
video_duration = total_frames / fps
total_sequences = math.ceil(video_duration / sequence_duration)

# create the destination directory
os.makedirs('seqs', exist_ok=True)

# iterate through each sequence to extract frames
frames_remaining = total_frames
for sq in range(total_sequences):
    # create the sequence directory
    seq_dir = f'seq_{sq}'
    os.makedirs('seqs/' + seq_dir, exist_ok=True)

    # extract each frame in the sequence & save it
    frame_count = 0
    while frames_remaining > 0 and frame_count < frames_per_sequence:
        ret, frame = cap.read()    
        frame_filename = f"frame_{frame_count:02}.jpg"
        frame_path = 'seqs/' + seq_dir + '/' + frame_filename
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        frames_remaining -= 1

cap.release()