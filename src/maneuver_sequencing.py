# Naively assumes execution from the /data/heats/heat_1Zj_jAPToxI/rides/ride_0 directory
# Some things to do next here:
#  - Instead of doing this for ride_0, loop through all of the rides in /rides/
#  - Resolve note 20241018_1
#  - Add error checking
#  - Remove hardcoding & allow for command-line arguments
#       Specifically, this script should be runnable from the project directory (or anywhere) and
#       simply accept a heat video ID and be able to navigate & create all the required directories

# Note 20241018_1: We're using 1-second sequences that allow us to more granularly capture maneuvers, and
#   support the use of start & end timestamps for each manuever. However, this might cause the model
#   to learn that a small chunk of a maneuver (say, the bottom turn before a roundhouse cutback) *equals*
#   that maneuver, which is not what we want. We went this route because we didn't want to undercount
#   maneuvers and wanted to account for longer manuevers (a 6-second barrel vs a 2-second one) but there
#   might be a better way. Consider, say, a 2-3 second sequence duration that checks whether a maneuver
#   *overlaps* with that sequence, rather than one thing fully contains the other (sequence, maneuver).
#   This alternative ensures that most maneuver-containing sequences fully contain the maneuver, with
#   the only exceptions being particularly long maneuvers (barrels, maybe a crazy floater), so the model
#   properly learns what a maneuver looks like. Sequences need to be short enough, however, that they're
#   unlikely to contain multiple maneuvers.

import cv2, math, os
import pandas as pd
vid_name = '1Zj_jAPToxI_0'
vid_file   = vid_name + '.mp4'
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
# TODO: Rethink approach here. See note 20241018_1 at the top.
sequence_duration = 1
frames_per_sequence = int(sequence_duration * fps)
video_duration = total_frames / fps
total_sequences = math.ceil(video_duration / sequence_duration)

# read the human labels & convert the timestamps
labels_df = pd.read_csv(human_file)
labels_df['start'] = labels_df['start'].apply(timestamp_to_seconds)
labels_df['end'] = labels_df['end'].apply(timestamp_to_seconds)

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

    # determine if this sequence is contained within a maneuver based on timestamps
    # TODO: Rethink approach here. See note 20241018_1 at the top.
    maneuvers_in_seq = labels_df[(labels_df['start'] <= start_time) & (labels_df['end'] >= end_time)]
    if len(maneuvers_in_seq) > 0:
        maneuver_type = maneuvers_in_seq.iloc[0]['maneuver_id']  # Use first maneuver in case of overlap
    else:
        maneuver_type = 'Z'
    
    # create the sequence label
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