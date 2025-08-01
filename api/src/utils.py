import csv, cv2, json, math, os, torch
import platform, requests, subprocess
import pandas as pd
import torchvision.transforms.functional as F
from collections import Counter
from PIL import Image

data_dir = "../../data"

def set_device():
    """Set device to GPU if available, otherwise use CPU.
    
    Returns:
        torch.device: The selected device (mps, cuda, or cpu)
    """
    print('>  Configuring device...')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('  >  Using CUDA (NVIDIA GPU) acceleration')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print('  >  Using MPS (Apple Silicon GPU) acceleration')
    else:
        device = torch.device("cpu")
        print('  >  Using CPU')
    return device

def format_time(seconds):
    """Format time in seconds to hours, minutes, seconds.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string (e.g. "2h 35m 13.5s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"

def sequence_video_frames(video_path, output_dir, sequence_duration=2):
    """
    Extract frames from a video and organize them into sequence directories.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory where sequence directories will be created
        sequence_duration: Duration of each sequence in seconds
    
    Returns:
        dict with metadata about the extracted sequences:
        {
            'total_sequences': int,
            'sequence_duration': int,
            'fps': float,
            'video_duration': float,
            'sequence_paths': list of paths to each sequence directory
        }
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_sequence = int(sequence_duration * fps)
    video_duration = total_frames / fps
    total_sequences = math.ceil(video_duration / sequence_duration)
    
    # Iterate through each sequence to extract frames
    sequence_paths = []
    frames_remaining = total_frames
    
    for sq in range(total_sequences):
        # Create the sequence directory
        seq_name = f'seq_{sq}'
        seq_path = os.path.join(output_dir, seq_name)
        os.makedirs(seq_path, exist_ok=True)
        sequence_paths.append(seq_path)
        
        # Extract each frame in the sequence & save it
        frame_count = 0
        while frames_remaining > 0 and frame_count < frames_per_sequence:
            ret, frame = cap.read()
            if not ret:  # Handle case where video ends unexpectedly
                break
                
            frame_filename = f"frame_{frame_count:02}.jpg"
            frame_path = os.path.join(seq_path, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            frames_remaining -= 1
    
    # Close the video file
    cap.release()
    
    # Return metadata about the sequences
    return {
        'total_sequences': total_sequences,
        'sequence_duration': sequence_duration,
        'fps': fps,
        'video_duration': video_duration,
        'sequence_paths': sequence_paths
    }

def timestamp_to_seconds(timestamp):
    """Convert 'MM:SS' to seconds."""
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

def label_sequences_from_csv(sequences_metadata, labels_df, output_csv_path=None):
    """
    Label sequences based on timestamp data from a CSV file.
    
    Args:
        sequences_metadata: Output from sequence_video_frames
        labels_df: DataFrame with 'start', 'end', and 'maneuver_id' columns
        output_csv_path: Optional path to save the sequence labels
    
    Returns:
        DataFrame with sequence labels
    """
    # Ensure start/end are in seconds
    if labels_df is not None and 'start' in labels_df and not isinstance(labels_df['start'].iloc[0], (int, float)):
        labels_df['start'] = labels_df['start'].apply(timestamp_to_seconds)
        labels_df['end'] = labels_df['end'].apply(timestamp_to_seconds)
    
    # Create sequence labels
    seq_labels = []
    sequence_duration = sequences_metadata['sequence_duration']
    total_sequences = sequences_metadata['total_sequences']
    
    for sq in range(total_sequences):
        # Infer the start & end timestamps of this sequence
        start_time = sq * sequence_duration
        end_time = min((sq+1) * sequence_duration, sequences_metadata['video_duration'])
        
        # Default maneuver_id is 0 (no maneuver)
        maneuver_id = 0
        
        # If we have labels, determine if this sequence contains a maneuver
        if labels_df is not None:
            maneuvers_in_seq = labels_df[(labels_df['start'] < end_time) & (labels_df['end'] > start_time)]
            if len(maneuvers_in_seq) > 0:
                # Use first maneuver in case of overlap
                maneuver_id = maneuvers_in_seq.iloc[0]['maneuver_id']
        
        # Create the sequence label
        seq_labels.append([f"seq_{sq}", maneuver_id])
    
    # Create DataFrame with labels
    seq_labels_df = pd.DataFrame(seq_labels, columns=['sequence_id', 'label'])
    
    # Save to CSV if path provided
    if output_csv_path:
        seq_labels_df.to_csv(output_csv_path, index=False)
    
    return seq_labels_df 

def load_frames_from_sequence(seq_dir, transform=None, mode='dev', add_batch_dim=False):
    """Load and transform frames from a sequence directory based on mode settings."""
    SKIP_FREQ = 10  # skip every 10 frames in dev mode
    COLOR = "L"     # "L" mode is for grayscale in dev mode
    if mode == 'prod':
        SKIP_FREQ =  2
        COLOR = "RGB"
    MAX_LENGTH = 60 // SKIP_FREQ  # base max sequence length
    
    # Load each frame in the sequence directory
    frames = []
    for frame_idx, frame_file in enumerate(sorted(os.listdir(seq_dir))):
        # Skip frames based on mode settings
        if frame_idx % SKIP_FREQ != 0:
            continue

        frame_path = os.path.join(seq_dir, frame_file)
        image = Image.open(frame_path).convert(COLOR)
        
        # Apply transforms
        if transform:
            # Use transform if provided (useful for backward compatibility)
            image = transform(image)
        else:
            # Use our new aspect-ratio preserving transform
            image = preserve_aspect_ratio_transform(image, target_size=224)
            
        frames.append(image)

    # Pad or truncate frames
    frames = pad_sequence(frames, MAX_LENGTH)
    
    # Add batch dimension if needed (for inference)
    if add_batch_dim:
        frames = frames.unsqueeze(0)
        
    return frames

def pad_sequence(frames, max_length=60):
    """Pad or truncate a sequence of frames to the specified length."""
    num_frames = len(frames)
    if num_frames < max_length:
        # Pad with zero tensors to reach max_length
        padding = [torch.zeros_like(frames[0]) for _ in range(max_length - num_frames)]
        frames = frames + padding
    else:
        # Truncate to max_length if too long
        frames = frames[:max_length]
    return torch.stack(frames)

def preserve_aspect_ratio_transform(image, target_size=224):
    """
    Resize an image while preserving its aspect ratio, then pad to make it square.
    
    Args:
        image: PIL Image
        target_size: The size of the output square image
        
    Returns:
        A square tensor of size (target_size, target_size)
    """
    # Get original dimensions
    width, height = image.size
    
    # Calculate scaling factor to fit the entire image within target_size
    # We need to scale based on the larger dimension to ensure the entire image fits
    scale_factor = target_size / max(width, height)
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image while preserving aspect ratio
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a square black background
    new_image = Image.new(resized_image.mode, (target_size, target_size), color=0)
    
    # Calculate position to paste the resized image (center it)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Paste the resized image onto the black background
    new_image.paste(resized_image, (paste_x, paste_y))
    
    # Convert to tensor
    tensor_image = F.to_tensor(new_image)
    
    return tensor_image

def load_maneuver_taxonomy():
    """Load the maneuver taxonomy from CSV file.
    
    Returns:
        dict: Mapping of maneuver IDs to their names
    """
    taxonomy = {}
    with open(data_dir + '/maneuver_taxonomy.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            taxonomy[int(row['id'])] = row['name']
    return taxonomy

def save_class_distribution(class_distribution):
    """Save class distribution to a JSON file.
    
    Args:
        class_distribution: Counter object with class distribution
    """
    # Convert Counter to dict for JSON serialization
    dist_dict = dict(class_distribution)
    
    # Save to JSON file
    with open(data_dir + '/class_distribution.json', 'w') as f:
        json.dump(dist_dict, f, indent=2)

def load_class_distribution():
    """Load class distribution from JSON file.
    
    Returns:
        Counter: Class distribution if file exists, None otherwise
    """
    try:
        with open(data_dir + '/class_distribution.json', 'r') as f:
            dist_dict = json.load(f)
        # Convert string keys to integers
        return Counter({int(k): v for k, v in dist_dict.items()})
    except FileNotFoundError:
        return None

def get_total_samples_in_heats():
    """Get total number of samples in the heats directory.
    
    Returns:
        int: Total number of samples
    """
    total = 0
    heats_dir = data_dir + "/heats"
    
    # Walk through all subdirectories
    for _, _, files in os.walk(heats_dir):
        # Count sequence directories
        if any(f.endswith('.jpg') for f in files):
            total += 1
            
    return total

def distribution_outdated():
    """Check if class distribution needs to be recalculated.
    
    Returns:
        bool: True if distribution should be recalculated
    """
    # Try to load existing distribution
    saved_dist = load_class_distribution()
    if saved_dist is None:
        return True
        
    # Get current total samples
    current_total = get_total_samples_in_heats()
    
    # Compare with saved total
    saved_total = sum(saved_dist.values())
    
    return current_total != saved_total

def collate_variable_length_videos(batch):
    """
    Custom collate function for DataLoader to handle variable-length videos.
    Pads shorter videos to match the length of the longest video in the batch.
    
    Args:
        batch: List of (video, score) tuples where videos may have different lengths
        
    Returns:
        videos: Padded video tensor of shape [batch_size, max_frames, channels, height, width]
        scores: Score tensor of shape [batch_size]
    """
    # Separate videos and scores
    videos, scores = zip(*batch)
    
    # Get the maximum number of frames in this batch
    max_frames = max(video.shape[0] for video in videos)
    
    # Pad each video to max_frames
    padded_videos = []
    for video in videos:
        if video.shape[0] < max_frames:
            # Calculate padding needed
            pad_size = max_frames - video.shape[0]
            # Pad with zeros at the end
            padded_video = torch.cat([
                video,
                torch.zeros(pad_size, *video.shape[1:], device=video.device)
            ])
            padded_videos.append(padded_video)
        else:
            padded_videos.append(video)
    
    # Stack videos and scores into batches
    videos_batch = torch.stack(padded_videos)
    scores_batch = torch.stack(scores)
    
    return videos_batch, scores_batch

def get_ec2_instance_type():
    """
    Get the AWS EC2 instance type by querying the instance metadata service.
    
    Returns:
        str: Instance type (e.g., 'g5.xlarge', 't3.medium') or None if not on EC2
    """
    try:
        response = requests.get('http://169.254.169.254/latest/meta-data/instance-type', timeout=1)
        return response.text
    except:
        return None
    
def get_machine_info():
    """
    Get a human-readable description of the current machine.
    
    Detects:
        - AWS EC2 instances (e.g., 'AWS g5.xlarge')
        - macOS machines with model info (e.g., 'macOS MacBookAir10,1')
        - Generic Linux/Windows systems
    
    Returns:
        str: Machine description (e.g., 'AWS g5.xlarge', 'macOS MacBookAir10,1', 'Linux x86_64')
    """
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin":  # macOS
        try:
            # Get Mac model using sysctl
            result = subprocess.run(['sysctl', '-n', 'hw.model'], capture_output=True, text=True)
            mac_model = result.stdout.strip()
            return f"macOS {mac_model}"
        except:
            return f"macOS {machine}"
    
    elif system == "Linux":
        try:
            # Try to get AWS instance type first
            ec2_type = get_ec2_instance_type()
            if ec2_type:
                return f"AWS {ec2_type}"
            
            # Fallback to generic Linux info
            return f"Linux {machine}"
        except:
            return f"Linux {machine}"
    
    else:
        return f"{system} {machine}"