import cv2, math, os, torch
import pandas as pd
from PIL import Image

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
        SKIP_FREQ =  1
        COLOR = "RGB"
    MAX_LENGTH = 60 / SKIP_FREQ  # base max sequence length
    
    # Load each frame in the sequence directory
    frames = []
    for frame_idx, frame_file in enumerate(sorted(os.listdir(seq_dir))):
        # Skip frames based on mode settings
        if frame_idx % SKIP_FREQ != 0:
            continue

        frame_path = os.path.join(seq_dir, frame_file)
        image = Image.open(frame_path).convert(COLOR)
        if transform:
            image = transform(image)
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