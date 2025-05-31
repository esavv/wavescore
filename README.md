# WaveScore

## Overview

This application will allow users to upload videos of themselves surfing and get their rides scored from 0 to 10 as if they're in a [surf competition](https://en.wikipedia.org/wiki/World_Surf_League#Judging[27]). Development is still in progress.

## Demo

[Watch it here!](https://www.youtube.com/shorts/CNARgUQ5YtU)

## Roadmap

### Things to Work on Next
Last updated: 2025/05/31
   - [IN PROGRESS] Build score prediction model: Infer wave score from raw video
   - Build web app as a client of the model inference API
   - Consider generating progressive score prediction: show user how predicted score changes as video progresses
   - Consider migrating maneuver prediction to TCN architecture to predict sequence of maneuvers from single video
   - Consider streamlining data labeling workflow & updating maneuver taxonomy for falls / failed moves
   - Migrate data from directory system to postgres + blob storage (S3)
   - API cleanup: Raise appropriate errors in api/video_content and api/video_overlay if key files are missing
   - API cleanup: Remove the Google Cloud base64 account key if no longer necessary
   - iOS code cleanup: Refactor toast & other logic in ContentView

### Things Recently Completed
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

## Admin Documentation

### Start & manage virtual environments when testing locally
Create a virtual environment
```bash
python3 -m venv venv
```

Activate the virtual environment
```bash
source venv/bin/activate
```

Deactivate it when done with the current session
```bash
deactivate
```

If the command prompt gets messed up after deactivating
```bash
export PS1="\h:\W \u$ "
```

### Download a YouTube video from command line using yt-dlp, ensure it's a mp4
```bash  
yt-dlp -f "mp4" -o "%(id)s.mp4" https://www.youtube.com/watch?v=1Zj_jAPToxI
```

### Download only specific subset of a longer YouTube video (from 30:00 to 1:00:00 in this example)
```bash  
yt-dlp -f "mp4" -o "%(id)s.mp4" --download-sections "*00:30:00-01:00:00" https://www.youtube.com/watch?v=1Zj_jAPToxI
```

### Download a YouTube video with script (recommended)
From the `/wavescore/src` directory, run:
```bash
python3 download_youtube.py <video_id>
```
This script downloads the video (full or partial) and creates the required directory structure in `/data/heats/heat_<video_id>/` with the video file and a CSV template for ride times.

### Clip a shorter video from a longer video and save it
This assumes we're in the parent `/wavescore` directory & naively saves it there.
```bash  
ffmpeg -i data/heats/heat_1Zj_jAPToxI/1Zj_jAPToxI.mp4 -ss 00:00:17 -to 00:00:46 -c:v libx264 -c:a aac 1Zj_jAPToxI_1.mp4
```

### Convert a longer surfing video into a sequence of ride clips
   - Suppose we have video `123.mp4`. First, ensure it exists at this path: `/wavescore/data/heats/123/123.mp4`
   - Add a csv to the `/123` directory called `ride_times.csv` that contains the start & end timestamps for each ride to be clipped
   - From the main `/wavescore` directory, run this command:
```bash  
python3 src/clipify.py 123 
```
   - This runs a python script that outputs the desired clips to this directory: `/wavescore/data/heats/heat_123/rides/`

### Convert a sequence of ride clips into labeled sequences of frames
This labeled sequence is intended to be fed into a model that will learn surf maneuvers from an input video.
   - Suppose we have video `123.mp4` that has ride clips in `/data/heats/123/rides/`
   - Ensure that each ride directory, in addition to the ride clip (e.g. `0/123_0.mp4`) has a human-labeled CSV file named `human_labels.csv` containing the start & end times of each maneuver performed in the ride, as well as the corresponding maneuver ID (see `data/maneuver_taxonomy.csv`)
   - From the main /wavescore directory, run this command:
```bash
python3 src/maneuver_sequencing.py 123
```
 - This runs a script that outputs frame sequences for each ride in, for example, `.../0/seqs` and outputs sequence labels in, for example, `.../0/seq_labels.csv`

### Time model training & inference runs for performance evaluation

Run these commands from the `/src` directory:
```bash
{ time python train.py --mode dev ; } 2> ../logs/train_time_$(date +"%Y%m%d_%H%M%S").log
{ time python inference.py --mode dev ; } 2> ../logs/inference_time_$(date +"%Y%m%d_%H%M%S").log
```

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
   - Create temporary branch heroku-main:
```bash
git checkout -b heroku-main
```
   - Un-ignore API files in .gitignore needed for deployment (remove these lines):
```
api/keys/
api/models/
```
   - Add & commit the changes to the heroku-main branch
```bash
git add -A
git commit -m 'Prep for Heroku deployment'
```
   - Use git subtrees to deploy updated API to Heroku:
```bash
git subtree push --prefix api heroku main
```
   - If there are any issues with divergent branches between local & remote, there's an ugly workaround for now: Reset the remote git repo and then try again:
```bash
heroku repo:reset --app surfjudge-api
```   
   - The better way to do this probably involves *not* creating a new heroku-main branch every time, but checking out a persistent one and some how fast-forwarding or rebasing it to the lastest commit in our main branch... to be investigated some other time
   - If there are issues with slug size, consider clearing Heroku build cache:
```bash
heroku plugins:install heroku-builds
heroku builds:cache:purge -a surfjudge-api
```
   - Check Heroku logs if needed:
```bash
heroku logs --tail --app surfjudge-api
```
   - Call the API from the command line to test:
```bash
curl -X POST https://surfjudge-api-71248b819ca4.herokuapp.com/upload_video -F "file=@data/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4"
```
   - Before switching back to main locally, re-ignore unignored files:
```bash
git rm --cached api/keys/*
git rm --cached api/models/*
```
   - Before switching back to main locally, re-add ignored files to .gitignore:
```
api/keys/
api/models/
```
   - Add & commit the changes to the heroku-main branch
```bash
git add -A
git commit -m 'Post-deployment return to main'
```
   - Switch back to main & delete the temp heroku branch
```bash
git checkout main
git branch -D heroku-main
```

### AWS EC2 Management - Training & Inference Servers
   - Instance choices:
      - Training server:  `g5.xlarge`
      - Inference server: `t3.medium`
   - SSH into an EC2 instance to manage my training and/or inference servers. From root dir:
```bash  
ssh -i "src/keys/aws_ec2.pem" ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com
```
   - From EC2 instance, make necessary project directories
```bash  
mkdir wavescore; cd wavescore
```
   - ...for training server
```bash
mkdir data logs models src
```
   - ...for inference server
```bash
mkdir data models src
```
   - Training server: Zip my src/ files to scp to AWS later
```bash  
zip -r src.zip augment_data.py checkpoints.py clipify.py create_maneuver_compilations.py dataset.py download_youtube.py inference.py maneuver_sequencing.py model_logging.py model.py requirements.txt train.py utils.py
```
   - Training server: Transfer my src zip to fresh EC2 instance from src dir:
```bash  
scp -i keys/aws_ec2.pem -r src.zip ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com:/home/ubuntu/wavescore/src
```
   - Training server: Zip my data/ files to scp to AWS later (lazy approach)
```bash  
zip -r data.zip heats/ inference_vids/ class_distribution.json maneuver_taxonomy.csv
```
   - Training server: Transfer my data zip to EC2 instance from data dir (lazy approach):
```bash  
scp -i ../src/keys/aws_ec2.pem -r data.zip ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com:/home/ubuntu/wavescore/data
```
   - Inference server: Zip my src/ files to scp to AWS later
```bash  
zip -r api.zip apidata/ keys/ app.py verify_video.py modify_video.py inference.py model.py utils.py checkpoints.py requirements_cpu.txt
```
   - Inference server: Transfer my api zip, taxonomy, and model to EC2 instance from src dir:
```bash  
scp -i keys/aws_ec2.pem -r api.zip ubuntu@ec2-3-88-165-100.compute-1.amazonaws.com:/home/ubuntu/wavescore/src
scp -i keys/aws_ec2.pem -r ../data/maneuver_taxonomy.csv ubuntu@ec2-3-88-165-100.compute-1.amazonaws.com:/home/ubuntu/wavescore/data
scp -i keys/aws_ec2.pem -r ../models/surf_maneuver_model_20250518_2118.pth ubuntu@ec2-3-88-165-100.compute-1.amazonaws.com:/home/ubuntu/wavescore/models
```
   - From EC2 instance, create a venv & install the required packages
```bash  
python3 -m venv venv
source venv/bin/activate
```
   - Training server install:
```bash  
pip install -r requirements.txt
```
   - Inference server install:
```bash  
pip install -r requirements_cpu.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
```
   - Inference server: Follow [ngrok linux installation instructions](https://dashboard.ngrok.com/get-started/setup/linux)
   - From EC2 instance, check disk space
```bash  
df -h
```
   - Training server: Copy model and training log from EC2 instance back to local project; run this locally
```bash  
scp -i keys/aws_ec2.pem ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com:'/home/ubuntu/wavescore/models/surf_maneuver_model_20250518_1431.pth' ../models/
scp -i keys/aws_ec2.pem ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com:'/home/ubuntu/wavescore/logs/training_20250518_1431.log' ../logs/
```


## Acknowledgments

TODO
