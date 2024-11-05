# Assumes execution from the /surfjudge parent project directory. Example execution:
#  > python3 src/maneuver_sequencing.py 1Zj_jAPToxI
#
# Some things to do next here:
#  - Add error checking
import cv2, math, os, sys
import pandas as pd

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
# assert that the /rides/ directory exists
rides_path = os.path.join(heat_path, 'rides')
if not os.path.exists(rides_path):
    print('Rides directory doesn\'t exist for heat ' + vid_id)
    sys.exit()

rides = [d for d in os.listdir(rides_path) if os.path.isdir(os.path.join(rides_path, d))]
rides.sort()

# helper function to convert 'MM:SS' to seconds
def timestamp_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

# iterate through each ride
for ride in rides:
    # extract the directory # (eg the 0 in ride_0)
    ride_no =  ride.split('_')[1]  

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

        # open the video file
        cap = cv2.VideoCapture(mp4_file)

        # get the fps, frame count, video duration, & number of sequences needed
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sequence_duration = 2
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

            # determine if this sequence contains or overlaps with a maneuver based on timestamps
            maneuvers_in_seq = labels_df[(labels_df['start'] < end_time) & (labels_df['end'] > start_time)]
            if len(maneuvers_in_seq) > 0:
                maneuver_id = maneuvers_in_seq.iloc[0]['maneuver_id']  # Use first maneuver in case of overlap
            else:
                maneuver_id = 0
            
            # create the sequence label
            seq_labels.append([f"seq_{sq}", maneuver_id])

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