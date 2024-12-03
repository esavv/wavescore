# SurfJudge

## Overview

[Development still in progress] This application will allow users to upload videos of themselves surfing and get their rides scored from 0 to 10 as if they're in a surf competition.

## Demo

TODO

## Roadmap

### Things to Work on Next
Last updated: 2024/11/29
   - Model API: Figure out how to host our python inference code in the Flask API. User posts a video file to the API service and gets an inference result returned
   - iOS + API: Build a demo where the API modifies the user's video, annotated with "their" maneuvers, returns the video to the app, the app plays the video and allows the user to save it to their device 
   - We should update all src/ scripts to expect execution from the src directory
   - Figure out how to pad images to make them square before the resizing in train.py

### Things Recently Completed
   - We can now clip full surf heats into individual rides, and process individual rides into maneuver-labeled sequences of frames that are ready to be fed into a model
   - We've finished labeling maneuvers for all rides in our first video
   - We've written initial code to convert our labeled sequences into a dataset to be fed into a model, structure & train a model on this dataset (dataset.py, model.py, train.py)
   - We've gotten train.py to build a model in dev mode, and should focus on getting inference to work now too
   - We have a basic inference script (inference.py) working and it can run inference on a target video (this is just one of the clips used in training, so not ideal). It doesn't work very well, it predicts "no maneuver" for every sequence in the video, even though several maneuvers are performed. But it at least makes predictions.
   - We have a boilerplate iOS app, we modified it to allow the user to upload a video, we gave it an app icon, and we figured out how to deploy and test it to my physical iPhone
   - iOS app: Add functionality that, upon user upload, displays a set of manuevers "performed" in the video. This should be a hardcoded set of maneuvers but gets us to figure out both (1) display and (2) what the json/schema the app should expect from our prediction service
   - Model API pre-work & iOS connection: Get a dummy API service running, deployed via Flask, and connect it to the iOS app. The service doesn't require/use video; it just returns the same hardcode set of maneuvers previously set in the iOS app code.
   - iOS app: The app now explicitly requests access to the user's videos when first installed & run
   - Model API: Deploy the Flask API to the cloud (via Heroku) and connect it to the iOS app. Configure the API to receive a video
   - iOS app: Update the iOS app to send the user's selected video. The API returns both a hardcoded set of maneuvers and the video's duration to prove it can actually use the video file. The results are displayed to the user in the app.
   - iOS app: Build a loading screen while waiting for API results

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
