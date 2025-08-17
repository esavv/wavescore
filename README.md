# WaveScore

## Overview

This application allows you to upload surfing videos and get your ride scored from 0 to 10 as if you're in a [surf competition](https://en.wikipedia.org/wiki/World_Surf_League#Judging[28]).

## Check it out!

Try it here: [wavescore.xyz](https://www.wavescore.xyz/)

... or [watch a demo](https://www.youtube.com/shorts/CNARgUQ5YtU)!

## Roadmap

### Current & Upcoming Tasks
Last updated: 2025/08/02
   - [IN PROGRESS] Build web app as a client of the model inference API
   - Cleanup: Organize `src` files into subdirectories
   - Cleanup: Refactor `src` to use filepaths relative to the absolute path for the main directory
   - Migrate maneuver prediction to TCN architecture to predict sequence of maneuvers from single video
   - Streamline data labeling workflow & updating maneuver taxonomy for falls / failed moves
   - Scale training data to improve predictions
   - Scale API servers to allow concurrent users
   - Generate progressive score prediction: show user how predicted score changes as video progresses
   - Migrate data from directory system to postgres + blob storage (S3)

### Completed Milestones
   - [2025/06/02] Build score prediction model: Infer wave score from raw video
   - [2025/05/20] Model training & inference deployed on AWS, Heroku abandoned
   - [2025/05/17] Maneuver prediction model works pretty well on training data (switched to 3D CNN model, data augmentation, power75 class weight correction, pretrained model layer freezing)
   - [2025/05/01] Model optimization: Pad video frames to make them square before the resizing in train.py
   - [2025/04/30] Refactoring: Refactor redundant code across inference.py, maneuver_sequencing.py, and dataset.py into common functions in utils.py
   - [2025/04/17] Switch API to SSE to display upload progress to user (checking if surf video, analyzing, annotating video...).
   - [2025/01/02] Revert API to hardcoded results after failing to deploy model inference (3GB+ in size) to Heroku
   - [2024/12/06] iOS + API: Demo where the API annotates the user's video file with hardcoded maneuvers, returns the video to the app, the app plays the video and allows the user to save it to their device
   - [2024/12/02] iOS + API: Check whether uploaded video is actually a surf video
   - [2024/11/29] iOS app: Display a loading screen while waiting for API results
   - [2024/11/29] Model API: Flask API deployed to the cloud (via Heroku) and connected to the iOS app. Configure the API to receive a video
   - [2024/11/21] iOS app: App sends the user's selected video to the API, which returns both hardcoded maneuvers and the video's duration to prove it can actually use the video file. Results are displayed to the user in the app.
   - [2024/11/20] iOS app: Explicitly request access to the user's videos when first installed & run
   - [2024/11/17] Model API pre-work & iOS connection: Dummy API service running locally, deployed via Flask, and connected to the iOS app. The service doesn't require/use video; it just returns the same hardcoded set of maneuvers previously set in the iOS app code.
   - [2024/11/12] iOS app: Upon user's video upload, display a set of manuevers "performed" in the video. This is a hardcoded set of maneuvers for now
   - [2024/11/08] Boilerplate iOS app modified to allow the user to upload a video, it has an app icon, and we can deploy and test it to my physical iPhone
   - [2024/11/06] Basic inference script (inference.py) works and can run inference on a target video (this is just one of the clips used in training, so not ideal). It doesn't work very well, it predicts "no maneuver" for every sequence in the video, even though several maneuvers are performed. But it at least makes predictions.
   - [2024/11/06] Build a model in dev mode (train.py)
   - Convert labeled frame sequences into a dataset to be fed into a model, structure & train a model on this dataset (dataset.py, model.py, train.py)
   - Label maneuvers for all rides in our first video
   - Clip full surf heats into individual rides, and process individual rides into maneuver-labeled sequences of frames that are ready to be fed into a model
