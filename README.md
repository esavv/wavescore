# WaveScore

## Overview

[Development still in progress] This application will allow users to upload videos of themselves surfing and get their rides scored from 0 to 10 as if they're in a surf competition.

## Demo

[Watch it here!](https://www.youtube.com/shorts/CNARgUQ5YtU)

## Roadmap

### Things to Work on Next
Last updated: 2024/12/10
   - Record demo recording and embed in this readme
   - Get current 2-part model inference running on API and returning results to iOS app
   - Build 1-part model: Infer score from raw video, no intermediate maneuver labeling
   - Migrate data from directory system to postgres + blob storage (S3)
   - Get interim analysis results (checking if surf video, analyzing, annotating video...). Consider switching API to SSE
   - Host model training & inference in the cloud
   - Pad video frames to make them square before the resizing in train.py
   - Build 2nd part of 2-part model: Infer wave score from maneuvers performed
   - iOS code cleanup: Refactor toast & other logic in ContentView
   - API code cleanup: Clearer naming for API keys and move them to dedicated subdirectory

### Things Recently Completed
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

## Admin Documentation

### Download a YouTube video using yt-dlp
```bash  
yt-dlp https://www.youtube.com/watch?v=1Zj_jAPToxI
```

### Clip a shorter video from a longer video and save it
This assumes we're in the parent `/surfjudge` directory & naively saves it there.
```bash  
ffmpeg -i data/heats/heat_1Zj_jAPToxI/1Zj_jAPToxI.mp4 -ss 00:00:17 -to 00:00:46 -c:v libx264 -c:a aac 1Zj_jAPToxI_1.mp4
```

### Convert a longer surfing video into a sequence of ride clips
   - Suppose we have video `123.mp4`. First, ensure it exists at this path: `/surfjudge/data/heats/heat_123/123.mp4`
   - Add a csv to the `/heat_123` directory called `ride_times_123.csv` that contains the start & end timestamps for each ride to be clipped
   - From the main `/surfjudge` directory, run this command:
```bash  
python3 src/clipify.py 123 
```
   - This runs a python script that outputs the desired clips to this directory: `/surfjudge/data/heats/heat_123/rides/`

### Convert a sequence of ride clips into labeled sequences of frames
This labeled sequence is intended to be fed into a model that will learn surf maneuvers from an input video.
   - Suppose we have video `123.mp4` that has ride clips in `/data/heats/heat_123/rides/`
   - Ensure that each ride directory, in addition to the ride clip (e.g. `ride_0/123_0.mp4`) has a human-labeled CSV file containing the start & end times of each maneuver performed in the ride, as well as the corresponding maneuver ID (see `data/maneuver_taxonomy.csv`)
   - From the main /surfjudge directory, run this command:
```bash
python3 src/maneuver_sequencing.py 123
```
 - This runs a script that outputs frame sequences for each ride in, for example, `.../ride_0/seqs` and outputs sequence labels in, for example, `.../ride_0/123_0_seq_labels.csv`

### Test Flask API locally and call it remotely via ngrok
   - Launch the Flask server:
```bash  
python3 app.py
```
   - Ensure environment variables are set correctly:
```bash  
export GOOGLE_APPLICATION_CREDENTIALS_B64=$(cat api/service_account_key.json.b64)
```
   - Launch the ngrok server:
```bash  
ngrok http 5000
```
   - Call the API:
```bash  
curl -X POST https://7c64-70-23-3-136.ngrok-free.app/upload_video -F "file=@tmp/IMG_1546.MOV"
```
   - If there are issues calling `localhost:5000` but not `127.0.0.1:5000`, it's because Apple AirPlay Receiver is listening on port 5000. While testing, disable it by navigating to: System Settings > General > AirDrop & Handoff > AirPlay Receiver (requires password to change)

### Update & deploy Flask API to Heroku
   - Switch to a development branch & make changes
   - Push changes to remote origin (on GitHub); merge to main remotely
   - Switch to main & pull from origin
   - Ensure environment variables are set correctly
```bash
heroku config:set GOOGLE_APPLICATION_CREDENTIALS_B64=$(cat api/service_account_key.json.b64)
heroku config:set AWS_ACCESS_KEY_ID=[INSERT_KEY_HERE]
heroku config:set AWS_SECRET_ACCESS_KEY=[INSERT_KEY_HERE]
```
   - Use git subtrees to deploy updated API to Heroku:
```bash
git subtree push --prefix api heroku main
```
   - Check Heroku logs if needed:
```bash
heroku logs --tail --app surfjudge-api
```
   - Call the API from the command line to test:
```bash
curl -X POST https://surfjudge-api-71248b819ca4.herokuapp.com/upload_video -F "file=@data/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4"
```

## Acknowledgments

TODO
