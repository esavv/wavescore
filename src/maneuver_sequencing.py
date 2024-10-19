# Naively assumes execution from the /data/heats/heat_1Zj_jAPToxI/rides/ directory
# Some things to do next here:
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
current_dir = os.getcwd()
rides = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
rides.sort()
vid_id = '1Zj_jAPToxI'

# helper function to convert 'MM:SS' to seconds
def timestamp_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

# iterate through each ride
for ride in rides:
    # extract the directory # (eg the 0 in ride_0)
    ride_no =  ride.split('_')[1]  

    # check if the video and human labels files are present
    ride_path = os.path.join(current_dir, ride)
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

        # open the video file
        cap = cv2.VideoCapture(mp4_file)

        # get the fps, frame count, video duration, & number of sequences needed
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # TODO: Rethink approach here. See note 20241018_1 at the top.
        sequence_duration = 1
        frames_per_sequence = int(sequence_duration * fps)
        video_duration = total_frames / fps
        total_sequences = math.ceil(video_duration / sequence_duration)

        # read the human labels & convert the timestamps
        labels_df = pd.read_csv(csv_file)
        labels_df['start'] = labels_df['start'].apply(timestamp_to_seconds)
        labels_df['end'] = labels_df['end'].apply(timestamp_to_seconds)

        # create the destination directory
        seqs_path = os.path.join(ride_path, 'seqs')
        os.makedirs(seqs_path, exist_ok=True)

        # iterate through each sequence to extract frames
        seq_labels = []
        frames_remaining = total_frames
        for sq in range(total_sequences):
            # create the sequence directory
            seq_path = os.path.join(seqs_path, f'seq_{sq}')
            os.makedirs(seq_path, exist_ok=True)

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
                frame_path = os.path.join(seq_path, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
                frames_remaining -= 1

        # save the sequence labels
        labels_file = vid_id + "_" + ride_no + '_seq_labels.csv'
        labels_path = os.path.join(ride_path, labels_file)
        seq_labels_df = pd.DataFrame(seq_labels, columns=['sequence_id', 'label'])
        seq_labels_df.to_csv(labels_path, index=False)
        
        cap.release()