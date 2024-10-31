# Things to work on next as of 2024/10/31:
 - We can now clip full surf heats into individual rides, and process individual rides into
  maneuver-labeled sequences of frames that are ready to be fed into a model
 - We've finished labeling maneuvers for all rides in our first video
 - We've written initial code to convert our labeled sequences into a dataset to be fed into a model, 
   structure & train a model on this dataset (dataset.py, model.py, train.py)
 - We should test train.py for errors
 - We have a basic script for trialing inference of our model, but it needs some serious work:
    - It needs to receive an input model as a command-line argument
    - It expects that the surf ride clip we're gonna run inference on has already been split into sequences
    - We need to identify a target clip to run inference on, save it somewhere, and write a separate
         script for converting it to frame sequences. We should reuse code from maneuver_sequencing.py
    - It should probably receive the target clip (or clip sequences) as a command-line argument
 - We should build a very simple iPhone app interface and figure out how to connect it to
    the model

## How to download a YouTube video using yt-dlp:
 > yt-dlp https://www.youtube.com/watch?v=1Zj_jAPToxI

## How to clip a shorter video from a longer video and save it to the appropriate directory:
   # Assumes we're in the parent /surfjudge directory & naively saves it here
 > ffmpeg -i data/heats/heat_1Zj_jAPToxI/1Zj_jAPToxI.mp4 -ss 00:00:17 -to 00:00:46 -c:v libx264 -c:a aac 1Zj_jAPToxI_1.mp4

## How to convert a longer surfing video into a sequence of ride clips:
 - Suppose we have video 123.mp4. First, ensure it exists at this path:
   > /surfjudge/data/heats/heat_123/123.mp4
 - Add a csv to the /heat_123 directory called 'ride_times_123.csv' that contains the start &
    end timestamps for each ride to be clipped
 - From the main /surfjudge directory, run this command:
   > python3 src/clipify.py 123
 - This runs a python script that outputs the desired clips to this directory:
   > /surfjudge/data/heats/heat_123/rides/

## How to convert a sequence of ride clips into labeled sequences of frames to be fed into a
    model that will learn surf maneuvers from an input video:
 - Suppose we have video 123.mp4 that has ride clips in /data/heats/heat_123/rides/
 - Ensure that each ride directory, in addition to the ride clip (eg ride_0/123_0.mp4) has
    a human-labeled CSV file containing the start & end times of each maneuver performed
    in the ride, as well as the corresponding maneuver ID (see data/maneuver_taxonomy.csv)
 - From the main /surfjudge directory, run this command:
   > python3 src/maneuver_sequencing.py 123
 - This runs a script that outputs frame sequences for each ride in, for example, .../ride_0/seqs
    and outputs sequence labels in, for example, .../ride_0/123_0_seq_labels.csv
