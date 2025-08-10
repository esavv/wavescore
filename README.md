# WaveScore

## Overview

This application allows you to upload surfing videos and get your ride scored from 0 to 10 as if you're in a [surf competition](https://en.wikipedia.org/wiki/World_Surf_League#Judging[27]).

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

## Admin Documentation

### Download a YouTube video from command line using yt-dlp, ensure it's a mp4
```bash  
yt-dlp -f "mp4" -o "%(id)s.mp4" https://www.youtube.com/watch?v=1Zj_jAPToxI
```

### Download only specific subset of a longer YouTube video (from 30:00 to 1:00:00 in this example)
```bash  
yt-dlp -f "mp4" -o "%(id)s.mp4" --download-sections "*00:30:00-01:00:00" https://www.youtube.com/watch?v=1Zj_jAPToxI
```

### Download a YouTube video with script
From the `./api/src` directory, run:
```bash
python3 download_youtube.py <video_id>
```
This script downloads the video (full or partial) and creates the required directory structure in `./data/heats/heat_<video_id>/` with the video file and a CSV template for ride times.

### Clip a shorter video from a longer video and save it
```bash  
ffmpeg -i data/heats/heat_1Zj_jAPToxI/1Zj_jAPToxI.mp4 -ss 00:00:17 -to 00:00:46 -c:v libx264 -c:a aac 1Zj_jAPToxI_1.mp4
```
This assumes we're in the main project directory & naively saves it there.

### Convert a longer surfing video into a sequence of ride clips
   - Suppose we have video `123.mp4`. First, ensure it exists at this path: `./data/heats/123/123.mp4`
   - Add a csv to the `/123` directory called `ride_times.csv` that contains the start & end timestamps for each ride to be clipped
   - From the `api` directory, run:
```bash  
python3 clipify.py 123 
```
   - This runs a python script that outputs the desired clips to this directory: `./data/heats/heat_123/rides/`

### Convert a sequence of ride clips into labeled sequences of frames
This labeled sequence is intended to be fed into a model that will learn surf maneuvers from an input video.
   - Suppose we have video `123.mp4` that has ride clips in `./data/heats/123/rides/`
   - Ensure that each ride directory, in addition to the ride clip (e.g. `0/123_0.mp4`) has a human-labeled CSV file named `human_labels.csv` containing the start & end times of each maneuver performed in the ride, as well as the corresponding maneuver ID (see `./data/maneuver_taxonomy.csv`)
   - From the main /wavescore directory, run this command:
```bash
python3 src/maneuver_sequencing.py 123
```
 - This runs a script that outputs frame sequences for each ride in, for example, `.../0/seqs` and outputs sequence labels in, for example, `.../0/seq_labels.csv`

### Start & manage virtual environments when testing locally
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Deactivate it when done with the current session
deactivate

# If the terminal prompt gets messed up after deactivating
export PS1="\h:\W \u$ "
```

### Time model training & inference runs for performance evaluation

Run these commands from `./api/src`:
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

### AWS EC2 Management - Training & Inference Servers
   - Instance choices:
      - Training server:  `g5.xlarge`
      - Inference server: `t3.medium`
   - SSH into an EC2 instance to manage my training and/or inference servers. From root dir:
```bash  
ssh -i "api/keys/aws_ec2.pem" ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com
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
   - Training server: Zip my `src` files to scp to AWS later
```bash  
zip -r src.zip augment_data.py checkpoints.py clipify.py create_maneuver_compilations.py dataset.py download_youtube.py inference.py maneuver_sequencing.py model_logging.py model.py requirements.txt train.py utils.py
```
   - Training server: Transfer my src zip to fresh EC2 instance from `api` dir:
```bash  
scp -i keys/aws_ec2.pem -r src/src.zip ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com:/home/ubuntu/wavescore/api/src
```
   - Training server: Zip my data/ files to scp to AWS later (lazy approach)
```bash  
zip -r data.zip heats/ inference_vids/ class_distribution.json maneuver_taxonomy.csv
```
   - Training server: Transfer my data zip to EC2 instance from `data` dir (lazy approach):
```bash  
scp -i ../api/keys/aws_ec2.pem -r data.zip ubuntu@ec2-44-210-82-47.compute-1.amazonaws.com:/home/ubuntu/wavescore/data
```
   - Inference server: Zip my `src` files to scp to AWS later
```bash  
zip -r api.zip ../apidata/ ../keys/ app.py verify_video.py modify_video.py inference.py model.py score_inference.py score_model.py score_dataset.py utils.py checkpoints.py requirements_cpu.txt
```
   - Inference server: Transfer my api zip, taxonomy, and model to EC2 instance from `api` dir:
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

### Enable HTTPS for EC2 Instance with Nginx

To enable HTTPS for the Flask API running on an AWS EC2 instance:

1. **Set Up a Subdomain, DNS, and Modify Security Group to**
   - Create a subdomain (e.g., `api.wavescore.xyz`) in our DNS provider (Vercel) and point it to the EC2 instance's public IP using an A record.
   - In the AWS EC2 console, add an inbound security group rule with the following settings:
     - Type: Custom TCP
     - Port range: 5000
     - Source: 0.0.0.0/0
     - Description: Flask API

2. **Install Nginx and Certbot on EC2**
   - SSH into ec2 instance and run:
   ```bash
   sudo apt update
   sudo apt install nginx certbot python3-certbot-nginx -y
   ```

3. **Configure Nginx as a Reverse Proxy**
   - Create or edit `/etc/nginx/sites-available/flask-app`; open it in an editor:
   ```bash
   sudo nano /etc/nginx/sites-available/flask-app
   ```
   - Populate this file with the config at ./api/nginx.conf
   - Save and exit: Ctrl+O, enter, Ctrl+X
   - Enable the config and reload Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/flask-app /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

4. **Open Ports 80 and 443 in Security Group**
   - In the AWS EC2 console, add these inbound security group rules:
     - **HTTP Rule:**
       - Type: HTTP
       - Port range: 80
       - Source: 0.0.0.0/0
       - Description: Web traffic
     - **HTTPS Rule:**
       - Type: HTTPS
       - Port range: 443
       - Source: 0.0.0.0/0
       - Description: Secure web traffic

5. **Obtain and Install SSL Certificate with Certbot**
   - Run Certbot to automatically configure SSL for the domain:
   ```bash
   sudo certbot --nginx -d api.wavescore.xyz
   ```
   - Follow the prompts to complete the certificate installation.

6. **Update Web App Environment Variables**
   - Set the API base URL to use HTTPS and the subdomain:
   ```env
   VITE_API_BASE_URL=https://api.wavescore.xyz
   ```

### Run Gunicorn Server on EC2

To run the Flask API as a systemd service using Gunicorn on the EC2 instance:

1. **Create the Systemd Service File**
   - SSH into the EC2 instance and create the service file:
   ```bash
   sudo vi /etc/systemd/system/wavescore-api.service
   ```
   - Populate this file with the configuration from `./api/systemd/wavescore-api.service`
   - Note: The service uses gevent to ensure Gunicorn works properly with long-lived SSE API connections

2. **Set Up Logs Directory**
   ```bash
   mkdir -p /home/ubuntu/wavescore/api/src/logs
   chown ubuntu:ubuntu /home/ubuntu/wavescore/api/src/logs
   ```

3. **Service Management Commands**
   ```bash
   # Enable service to start on boot
   sudo systemctl enable wavescore-api

   # Reload systemd if you edit the service file
   sudo systemctl daemon-reload

   # Restart the service
   sudo systemctl restart wavescore-api

   # Check service status
   sudo systemctl status wavescore-api

   # Stop the service
   sudo systemctl stop wavescore-api
   ```

4. **Monitoring: API logs, events, and server memory**
   ```bash
   # Follow the error log in real time
   tail -f ~/wavescore/api/src/logs/error.log

   # Follow the access log in real time
   tail -f ~/wavescore/api/src/logs/access.log

   # Follow systemd service logs (includes start/stop, restarts, oom kills)
   journalctl -u wavescore-api -f

   # Watch memory usage to catch OOM pressure early
   watch -n1 free -h
   ```

### Web App Development & Deployment

To set up and deploy a React web app:

1. **Setup Vite & React**
   ```bash
   npm init vite@latest web
   cd web
   npm install
   ```

2. **Add Tailwind CSS**
   ```bash
   npm install -D tailwindcss@3 postcss autoprefixer
   npx tailwindcss init -p
   ```
   - In `tailwind.config.js`, set the `content` array:
   ```js
   export default {
     content: [
       "./index.html",
       "./src/**/*.{js,ts,jsx,tsx}",
     ],
     theme: {
       extend: {},
     },
     plugins: [],
   }
   ```
   - In `src/index.css`, replace all content with:
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   ```

3. **Add CSS Linting with Stylelint (Tailwind Compatible)**
   ```bash
   npm install -D stylelint stylelint-config-standard stylelint-config-tailwindcss
   ```
   - In the project root, create a `stylelint.config.cjs` file with:
   ```js
   module.exports = {
     extends: [
       "stylelint-config-standard",
       "stylelint-config-tailwindcss"
     ],
     rules: {}
   }
   ```

4. **Run Locally**
   ```bash
   npm run dev
   ```
   This starts the development server (usually at `http://localhost:5173`). The app will hot-reload as we make changes.

5. **Deploy with Vercel**
   - Go to [https://vercel.com](https://vercel.com)
   - Sign in with GitHub
   - Import the wavescore repo

   **For Monorepo Setup:**

   Since we're building this as part of a larger wavescore project, configure Vercel to deploy only the web subdirectory:

   - During Vercel setup, set **Root Directory** to `web/`
   - Set **Build Command** to `npm run build`
   - Set **Output Directory** to `dist` (Vite default)

   The project structure:
   ```
   wavescore/
     ├── api/
     ├── data/
     ├── web/         👈 Vercel will build from here
     ├── README.md
     └── .git/
   ```

   Vercel will only build and deploy from the `web/` folder, ignoring the rest of the monorepo.

   - Click **Deploy**
