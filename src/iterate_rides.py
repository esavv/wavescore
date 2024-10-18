# Placeholder for updates to maneuver_sequencing.py
# Assumes that we're in the /rides directory
import os
current_directory = os.getcwd()
rides = [d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]

for ride in rides:
    # extract the directory # (eg the 0 in ride_0)
    # open the directory
    # assert that the video is there (1Zj_jAPToxI_0.mp4)
    # assert that the human labels file is there (1Zj_jAPToxI_0_human_labels.csv)
    # if all is good, perform maneuver sequencing for this ride
    # if necessary, go back to the parent directory

    # delete this
    print(ride)