# Admin Documentation

## Download and manipulate YouTube videos
```bash  
# Download a YouTube video from command line using yt-dlp, ensure it's a mp4
yt-dlp -f "mp4" -o "%(id)s.mp4" https://www.youtube.com/watch?v=1Zj_jAPToxI

# Download only specific subset of a longer YouTube video (from 30:00 to 1:00:00 in this example)
yt-dlp -f "mp4" -o "%(id)s.mp4" --download-sections "*00:30:00-01:00:00" https://www.youtube.com/watch?v=1Zj_jAPToxI

# Download a YouTube video with script
# From the ./api/src directory, run:
python3 download_youtube.py <video_id>
```
   - The previous script downloads the video (full or partial) and creates the required directory structure in `./data/heats/heat_<video_id>/` with the video file and a CSV template for ride times.

```bash  
# Clip a shorter video from a longer video and save it
# This assumes we're in the main project directory & naively saves it there.
ffmpeg -i data/heats/heat_1Zj_jAPToxI/1Zj_jAPToxI.mp4 -ss 00:00:17 -to 00:00:46 -c:v libx264 -c:a aac 1Zj_jAPToxI_1.mp4
```

   - Convert a longer surfing video into a sequence of ride clips
     - Suppose we have video `123.mp4`. First, ensure it exists at this path: `./data/heats/123/123.mp4`
     - Add a csv to the `/123` directory called `ride_times.csv` that contains the start & end timestamps for each ride to be clipped
```bash  
# From the `api` directory, run:
python3 clipify.py 123 
```
     - This runs a python script that outputs the desired clips to this directory: `./data/heats/heat_123/rides/`

   - Convert a sequence of ride clips into labeled sequences of frames
     - Suppose we have video `123.mp4` that has ride clips in `./data/heats/123/rides/`
     - Ensure that each ride directory, in addition to the ride clip (e.g. `0/123_0.mp4`) has a human-labeled CSV file named `human_labels.csv` containing the start & end times of each maneuver performed in the ride, as well as the corresponding maneuver ID (see `./data/maneuver_taxonomy.csv`)
```bash
# From the main /wavescore directory, run this command:
python3 src/maneuver_sequencing.py 123
```
   - This runs a script that outputs frame sequences for each ride in, for example, `.../0/seqs` and outputs sequence labels in, for example, `.../0/seq_labels.csv`

## Start & manage virtual environments when testing locally
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

## Time model training & inference runs for performance evaluation

Run these commands from `./api/src`:
```bash
{ time python train.py --mode dev ; } 2> ../logs/train_time_$(date +"%Y%m%d_%H%M%S").log
{ time python inference.py --mode dev ; } 2> ../logs/inference_time_$(date +"%Y%m%d_%H%M%S").log
```

## Test Flask API locally and call it remotely via ngrok
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

## AWS EC2 Management - Training & Inference Servers
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

## Enable HTTPS for EC2 Instance with Nginx

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

## Run Gunicorn Server on EC2

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

## Web App Development & Deployment

To set up and deploy a React web app:

1. **Set up Vite & React**
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

4. **Run develoment server locally**
   ```bash
   npm run dev
   ```

5. **Deploy with Vercel**
   - Go to [https://vercel.com](https://vercel.com)
   - Sign in with GitHub
   - Import the wavescore repo

   **For Monorepo Setup:**

   In the current project structure the web app is just one component, so configure Vercel to deploy only the web subdirectory:

   - During Vercel setup, set **Root Directory** to `web/`
   - Set **Build Command** to `npm run build`
   - Set **Output Directory** to `dist` (Vite default)

   The project structure:
   ```
   wavescore/
     ├── api/
     ├── data/
     ├── web/         <-- Vercel will build from here
     ├── README.md
     └── .git/
   ```

   Vercel will only build and deploy from the `web/` folder, ignoring the rest of the monorepo.

   - Click **Deploy**

## Launch a New Inference Server

1. Launch EC2 instance
   - AWS Console > EC2 > Instances > Launch instances
   - Name convention: wavescore-api-{instance_type} (e.g. wavescore-api-t3medium)
   - OS: Ubuntu
   - AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch (2.6, 2.7)
   - Select an instance type (e.g. t3.medium, g5.xlarge)
   - Key pair: esavage_ec2
   - Create default security group
   - Allow SSH traffic from anywhere
   - Default storage option

2. Load & install application 
 ```bash
 # Zip API files locally (run from the repo's ./api directory)
 # Create deploy bundle with API source, service files, nginx config, and CPU requirements
 zip -r api.zip \
   apidata/ \
   keys/ \
   nginx.conf \
   requirements.txt \
   requirements_cpu.txt \
   src/ \
   systemd/ 

  # Create remote directory structure to receive application files
  ssh -i keys/aws_ec2.pem ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com 'mkdir -p ~/wavescore/api ~/wavescore/data ~/wavescore/models' 

  # Transfer bundle, taxonomy, and model artifact(s) to the EC2 instance
  scp -i keys/aws_ec2.pem api.zip ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com:/home/ubuntu/wavescore/api
  scp -i keys/aws_ec2.pem ../data/maneuver_taxonomy.csv ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com:/home/ubuntu/wavescore/data
 
 # Replace the model filename(s) below with your actual filenames
  scp -i keys/aws_ec2.pem ../models/surf_maneuver_model_20250518_2118.pth ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com:/home/ubuntu/wavescore/models
  scp -i keys/aws_ec2.pem ../models/score_model_20250602_1643.pth ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com:/home/ubuntu/wavescore/models
 
  # SSH into EC2
  ssh -i keys/aws_ec2.pem ubuntu@ec2-XX-XX-XX-XX.compute-1.amazonaws.com
  ```

  ```bash
  # Unzip and organize API files
  cd ~/wavescore/api
  unzip api.zip

  # If necessary, install python
  sudo apt install python3.10-venv

  # Create and activate a virtual environment
  python3 -m venv venv
  source venv/bin/activate

  # Install dependencies
  # If this is a CPU instance:
  pip install -r requirements_cpu.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu

  # If this is a GPU instance:
  pip install -r requirements.txt

  # Deactivate virtual environment when done with installation
  deactivate

  # (Optional) Check disk space
  df -h

  # If main volume doesn't have enough space for dependencies, use NVMe:
  #   Remove old venv and pip caches on root to reclaim space
  rm -rf /home/ubuntu/wavescore/api/venv
  pip cache purge || true
  rm -rf ~/.cache/pip
  sudo rm -rf /tmp/pip-* /tmp/tmp.* 2>/dev/null || true

  #   Prepare NVMe dirs and permissions
  sudo mkdir -p /opt/dlami/nvme/{venvs,pip-cache,tmp}
  sudo chown -R ubuntu:ubuntu /opt/dlami/nvme

  #   Create new venv on NVMe and install requirements using NVMe cache/tmp
  python3 -m venv /opt/dlami/nvme/venvs/wavescore
  source /opt/dlami/nvme/venvs/wavescore/bin/activate
  pip install -U pip
  PIP_CACHE_DIR=/opt/dlami/nvme/pip-cache TMPDIR=/opt/dlami/nvme/tmp \
  pip install --no-cache-dir -r /home/ubuntu/wavescore/api/requirements.txt

  #   Ensure the gunicorn log directory exists relative to WorkingDirectory
  mkdir -p /home/ubuntu/wavescore/api/src/logs
 ```

3. Configure networking
 - Set up inbound security group rules
 - In EC2 > Instances > {your instance} > Security > Open security group
   - **HTTP**
     - Type: HTTP
     - Port range: 80
     - Source: 0.0.0.0/0
     - Description: Allow HTTP
   - **HTTPS**
     - Type: HTTPS
     - Port range: 443
     - Source: 0.0.0.0/0
     - Description: Allow HTTPS
 - Point API endpoint to new instance
 - In Vercel > project > Settings > Domains > api.wavescore.xyz > View DNS Records
   - Edit Value for Name `api` and set to new instance public IP address
 
 ```bash
 # Validate that domain now points to instance IP address
 dig +short api.wavescore.xyz

 # Install nginx and certbot
 sudo apt update
 sudo apt install -y nginx certbot python3-certbot-nginx
 
 # Obtain and install SSL certificate
 sudo certbot --nginx -d api.wavescore.xyz --agree-tos -m <YOUR_EMAIL_HERE> --non-interactive --redirect
 
 # Configure nginx as a reverse proxy
 # Copy the provided nginx config from the repo and enable it
 sudo cp ~/wavescore/api/nginx.conf /etc/nginx/sites-available/flask-app
 sudo ln -sf /etc/nginx/sites-available/flask-app /etc/nginx/sites-enabled/
 sudo nginx -t && sudo systemctl reload nginx

 # If any conflicting server names, check server names
 sudo ls -l /etc/nginx/sites-enabled

 # Remove any that aren't flask-app
 sudo rm -f /etc/nginx/sites-enabled/default
 ```

4. Launch API service
 ```bash
 # Create the systemd service file from the repo copy
 sudo cp ~/wavescore/api/systemd/wavescore-api.service /etc/systemd/system/wavescore-api.service

 # If using NVMe for virtual environment, use this instead
 sudo cp ~/wavescore/api/systemd/wavescore-api.service.nvme /etc/systemd/system/wavescore-api.service
 
 # Set up logs directory and permissions
 mkdir -p ~/wavescore/api/src/logs
 sudo chown ubuntu:ubuntu ~/wavescore/api/src/logs
 
 # Enable on boot, reload daemon, and start the service
 sudo systemctl daemon-reload
 sudo systemctl enable wavescore-api
 sudo systemctl restart wavescore-api
 
 # Verify status and follow logs
 sudo systemctl status wavescore-api
 tail -f ~/wavescore/api/src/logs/error.log
 ```
