# Things to work on next as of 2024/11/07:
 - We can now clip full surf heats into individual rides, and process individual rides into
  maneuver-labeled sequences of frames that are ready to be fed into a model
 - We've finished labeling maneuvers for all rides in our first video
 - We've written initial code to convert our labeled sequences into a dataset to be fed into a model, 
   structure & train a model on this dataset (dataset.py, model.py, train.py)
 - We've gotten train.py to build a model in dev mode, and should focus on getting inference to work now too
 - We have a basic inference script (inference.py) working and it can run inference on a target video
      (this is just one of the clips used in training, so not ideal). It doesn't work very well, it predicts
      "no maneuver" for every sequence in the video, even though several maneuvers are performed. But it at
      least makes predictions.
 - We should build a very simple iPhone app interface and figure out how to connect it to
    the model. We created a boilerplate project in Xcode and now need to start adding functionality.
 - We should update all src/ scripts to expect execution from the src directory
 - Figure out how to pad images to make them square before the resizing in train.py
 - Figure out how to get progress bar-like print statements

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
