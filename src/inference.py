import torch
from torchvision import transforms
from dataset import SurfManeuverDataset
from model import SurfManeuverModel
from PIL import Image
import argparse, cv2, math, os

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

def infer_sequence(model, seq_dir, mode='dev'):
    """Run inference on a single sequence."""
    sequence = load_sequence(seq_dir, mode=mode)
    sequence = sequence.to(device)
    with torch.no_grad():
        output = model(sequence)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

# Set up command-line arguments & configure 'prod' and 'dev' modes (via an environment variable).
print('Setting up command-line arguments')
parser = argparse.ArgumentParser(description='Toggle between prod and dev modes.')
parser.add_argument('--mode', choices=['prod', 'dev'], default='dev', help='Set the application mode (prod or dev).')
args = parser.parse_args()
mode = args.mode

# Set device to GPU if available, otherwise use CPU
print('Configuring device & model tranforms... (mode is: ' + mode + ')')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the saved model
print('Loading the model...')
model_path = "../models/surf_maneuver_model_20241106_1324.pth"
model = SurfManeuverModel(num_classes=10)  # Ensure num_classes matches your trained model
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the video target for inference
print('Loading target video...')
inference_target = "../data/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4"
target_path = os.path.dirname(inference_target)

# Convert the inference target to frame sequences
#   open the video file
print('Opening the video file & extracting frames...')
cap = cv2.VideoCapture(inference_target)

#   get the fps, frame count, video duration, & number of sequences needed
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
sequence_duration = 2
frames_per_sequence = int(sequence_duration * fps)
video_duration = total_frames / fps
total_sequences = math.ceil(video_duration / sequence_duration)

#   create the destination directory
seqs_path = os.path.join(target_path, 'seqs')
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
        ret, frame = cap.read()    
        frame_filename = f"frame_{frame_count:02}.jpg"
        frame_path = os.path.join(seq_path, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        frames_remaining -= 1
cap.release()

# Run inference on each sequence in the new video
print('Running inference...')
for sq in range(total_sequences):
# for seq_name in sorted(os.listdir(seqs_path)):
    seq_name = f'seq_{sq}'
    seq_dir = os.path.join(seqs_path, seq_name)
    if os.path.isdir(seq_dir):
        prediction = infer_sequence(model, seq_dir, mode=mode)
        print(f"  Predicted maneuver for {seq_name}: {prediction}")
