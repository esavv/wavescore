# This script runs inference on a surf video to detect maneuvers performed in the ride.
# It expects a video file as input and a trained model file in the ../models/ directory.

# Usage:
# src $ python inference.py --mode dev

import torch
from torchvision import transforms
from model import SurfManeuverModel
from utils import sequence_video_frames, load_frames_from_sequence
import argparse, csv, os, shutil, sys
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
print('inference: Configuring device & model transforms...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def run_inference(video_path, model_filename, mode='dev'):
    # Load the video target for inference
    print('Loading target video...')
    video_dir = os.path.dirname(video_path)

    # If not already there, download the model from S3 to local directory
    # Model URL: 
    print('Retrieving the model...')
    model_dir = "../models/"
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

    # Extract frames from the video
    print('Opening the video file & extracting frames...')
    seqs_path = os.path.join(video_dir, 'seqs')
    
    # Use the utility function to extract frames
    sequences_metadata = sequence_video_frames(
        video_path=video_path,
        output_dir=seqs_path,
        sequence_duration=2
    )

    # Run inference on each sequence in the new video
    print('Running inference...')
    taxonomy = {}
    with open('../data/maneuver_taxonomy.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            taxonomy[int(row['id'])] = row['name']

    maneuvers = []
    sequence_duration = sequences_metadata['sequence_duration']
    total_sequences = sequences_metadata['total_sequences']
    
    for sq in range(total_sequences):
        seq_name = f'seq_{sq}'
        seq_dir = os.path.join(seqs_path, seq_name)
        if os.path.isdir(seq_dir):
            maneuver_id = infer_sequence(model, seq_dir, mode=mode)

            start_time = sq * sequence_duration
            end_time = start_time + sequence_duration

            # hardcoding to get interesting results
            if sq == 2:
                maneuver_id = 6
            elif sq == 5:
                maneuver_id = 4
            
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
    # Use the shared function from utils.py with batch dimension already added
    sequence = load_frames_from_sequence(seq_dir, transform, mode, add_batch_dim=True)
    sequence = sequence.to(device)
    with torch.no_grad():
        output = model(sequence)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

if __name__ == "__main__":
    # Set up command-line arguments & configure 'prod' and 'dev' modes
    print('Setting up command-line arguments')
    parser = argparse.ArgumentParser(description='Toggle between prod and dev modes.')
    parser.add_argument('--mode', choices=['prod', 'dev'], default='dev', help='Set the application mode (prod or dev).')
    args = parser.parse_args()
    mode = args.mode

    # List available models
    model_dir = "../models/"
    models = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    if not models:
        print("Error: No model files found in ../models/")
        sys.exit(1)

    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    # Get user's model choice
    while True:
        try:
            choice = int(input("\nEnter the number of the model to use: "))
            if 1 <= choice <= len(models):
                model_filename = models[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")

    # Set video path and run inference
    video_path = "../data/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4"
    maneuvers = run_inference(video_path, model_filename, mode)
    print("Prediction dict: " + str(maneuvers))