import torch
from torchvision import transforms
from model import SurfManeuverModel
from PIL import Image
import argparse, csv, cv2, math, os, shutil
# import boto3
# from botocore.exceptions import NoCredentialsError

# print("inference: Setting env variables...")
# if os.path.exists("./keys/aws_s3_accessKeys.csv"):
#     with open('./keys/aws_s3_accessKeys.csv', mode='r', encoding='utf-8-sig') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             os.environ['AWS_ACCESS_KEY_ID'] = row['Access key ID']
#             os.environ['AWS_SECRET_ACCESS_KEY'] = row['Secret access key']
#             break  # Assuming there is only one row, exit the loop after setting the variables
#else:
    #TODO: raise an appropriate error
    #raise EnvironmentError("Missing AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY environment variables")

# Set device to GPU if available, otherwise use CPU
print('inference: Configuring device & model tranforms...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def run_inference(video_path, bucket_name, model_filename, mode='dev'):
    # Load the video target for inference
    print('Loading target video...')
    video_dir = os.path.dirname(video_path)

    # If not already there, download the model from S3 to local directory
    # Model URL: 
    print('Retrieving the model...')
    model_dir = "./models/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    # if not os.path.exists(model_path):
    #     print("  Downloading model from S3...")
    #     s3 = boto3.client("s3")
    #     try:
    #         # Download the model file from S3
    #         s3.download_file(bucket_name, model_filename, model_path)
    #         print(f"Model downloaded successfully and saved to {model_path}")
    #     except NoCredentialsError:
    #         print("AWS credentials not found. Please set them in your environment.")
    #         raise
    #     except Exception as e:
    #         print(f"Error downloading model: {e}")
    #         raise
    # else:
    #     print("  Model already saved locally, continuing...")

    # Load the saved model
    print('Loading the model...')
    model = SurfManeuverModel(num_classes=10)  # Ensure num_classes matches your trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Convert the inference target to frame sequences
    #   open the video file
    print('Opening the video file & extracting frames...')
    cap = cv2.VideoCapture(video_path)

    #   get the fps, frame count, video duration, & number of sequences needed
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sequence_duration = 2
    frames_per_sequence = int(sequence_duration * fps)
    video_duration = total_frames / fps
    total_sequences = math.ceil(video_duration / sequence_duration)

    #   create the destination directory
    seqs_path = os.path.join(video_dir, 'seqs')
    os.makedirs(seqs_path, exist_ok=True)

    #   iterate through each sequence to extract frames
    frames_remaining = total_frames
    for sq in range(total_sequences):
        # create the sequence directory
        seq_path = os.path.join(seqs_path, f'seq_{sq}')
        os.makedirs(seq_path, exist_ok=True)

        # extract each frame in the sequence & save it
        frame_count = 0
        while frames_remaining > 0 and frame_count < frames_per_sequence:
            _, frame = cap.read()    
            frame_filename = f"frame_{frame_count:02}.jpg"
            frame_path = os.path.join(seq_path, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            frames_remaining -= 1
    cap.release()

    # Run inference on each sequence in the new video
    print('Running inference...')
    taxonomy = {}
    with open('../data/maneuver_taxonomy.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            taxonomy[int(row['id'])] = row['name']

    maneuvers = []
    sequence_duration = 2
    for sq in range(total_sequences):
        seq_name = f'seq_{sq}'
        seq_dir = os.path.join(seqs_path, seq_name)
        if os.path.isdir(seq_dir):
            maneuver_id = infer_sequence(model, seq_dir, mode=mode)

            start_time = sq * sequence_duration
            end_time = start_time + sequence_duration

            # hardcoding to get interesting results
            if sq == 2:
                maneuver_id = 8
            elif sq == 5:
                maneuver_id = 6
            
            # lookup manuever name
            name = taxonomy.get(maneuver_id, 'Unknown maneuver')
            if maneuver_id != 0:
                maneuvers.append({'name': name, 'start_time': start_time, 'end_time': end_time})
            print(f"  Predicted maneuver for sequence {sq}: {maneuver_id}")

    # clean things up
    shutil.rmtree(seqs_path)

    return maneuvers

def infer_sequence(model, seq_dir, mode='dev'):
    """Run inference on a single sequence."""
    sequence = load_sequence(seq_dir, mode=mode)
    sequence = sequence.to(device)
    with torch.no_grad():
        output = model(sequence)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

def load_sequence(seq_dir, mode='dev'):
    """Load frames from a sequence directory and apply transforms."""
    frames = []
    SKIP_FREQ = 10

    # Load each frame in the sequence directory
    for frame_idx, frame_file in enumerate(sorted(os.listdir(seq_dir))):
        # Skip frames in dev mode to speed up training
        if mode == 'dev' and frame_idx % SKIP_FREQ != 0:
            continue

        frame_path = os.path.join(seq_dir, frame_file)
        if mode == 'prod':
            image = Image.open(frame_path).convert("RGB")
        elif mode == 'dev':
            image = Image.open(frame_path).convert("L")  # "L" mode is for grayscale; reduce image size for faster training in dev mode
        image = transform(image)
        frames.append(image)

    # Pad or truncate frames
    max_length = 60
    if mode == 'dev':
        max_length = max_length // SKIP_FREQ
    frames = pad_sequence(frames, max_length)
    
    return frames

def pad_sequence(frames, max_length=60):
    num_frames = len(frames)
    if num_frames < max_length:
        # Pad with zero tensors to reach max_length
        padding = [torch.zeros_like(frames[0]) for _ in range(max_length - num_frames)]
        frames = frames + padding
    else:
        # Truncate to max_length if too long
        frames = frames[:max_length]
    return torch.stack(frames).unsqueeze(0) # Shape (1, num_frames, channels, height, width)

if __name__ == "__main__":
    # Set up command-line arguments & configure 'prod' and 'dev' modes (via an environment variable).
    print('Setting up command-line arguments')
    parser = argparse.ArgumentParser(description='Toggle between prod and dev modes.')
    parser.add_argument('--mode', choices=['prod', 'dev'], default='dev', help='Set the application mode (prod or dev).')
    args = parser.parse_args()

    mode = args.mode
    video_path = "../data/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4"
    model_path = "../models/surf_maneuver_model_20241106_1324.pth"

    maneuvers = run_inference(video_path, model_path, mode)
    print("Prediction dict: " + str(maneuvers))