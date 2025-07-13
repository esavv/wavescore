# This script generates augmented versions of surf videos for training data.
# It applies various transformations (flips, rotations, brightness adjustments) to create
# additional training examples while preserving the original data structure.

# Usage:
#     # Generate all possible augmentations for all heats
#     python augment_data.py
#
#     # Generate only flipped versions of a specific heat
#     python augment_data.py --heat 1Zj_jAPToxI_6 --transform flip
#
#     # Remove all augmented heat directories
#     python augment_data.py --cleanup

import argparse, cv2, shutil
import numpy as np
from pathlib import Path

# Define valid transformations
VALID_TRANSFORMATIONS = {
    'flip': 'flipped',
    'rotate_pos': 'rot_pos5',
    'rotate_neg': 'rot_neg5',
    'brighten': 'bright_up',
    'darken': 'bright_down'
}

heats_path = '../../data/heats'

def load_frame_sequence(seq_dir):
    """Load all frames from a frame sequence directory."""
    frames = []
    for frame_file in sorted(Path(seq_dir).glob('*.jpg')):
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            frames.append(frame)
    return np.array(frames)

def save_frame_sequence(frames, output_dir):
    """Save frames as a sequence of jpg files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        output_path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(output_path), frame)

def load_video(video_path):
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def save_video(frames, output_path, fps=30):
    """Save a sequence of frames as a video file."""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()

def horizontal_flip(frames):
    """Flip all frames horizontally."""
    return np.flip(frames, axis=2)

def rotate_frames(frames, angle):
    """Rotate all frames by the specified angle."""
    rows, cols = frames[0].shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return np.array([cv2.warpAffine(frame, M, (cols, rows)) for frame in frames])

def adjust_brightness(frames, factor):
    """Adjust brightness of all frames."""
    hsv_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames])
    hsv_frames[..., 2] = np.clip(hsv_frames[..., 2] * factor, 0, 255)
    return np.array([cv2.cvtColor(frame, cv2.COLOR_HSV2BGR) for frame in hsv_frames])

def transform_frames(frames, transform_type):
    """Apply specified transformation to frames."""
    transform_name = VALID_TRANSFORMATIONS.get(transform_type)
    if transform_name == 'flipped':
        return horizontal_flip(frames)
    elif transform_name == 'rot_pos5':
        return rotate_frames(frames, 5)
    elif transform_name == 'rot_neg5':
        return rotate_frames(frames, -5)
    elif transform_name == 'bright_up':
        return adjust_brightness(frames, 1.2)
    elif transform_name == 'bright_down':
        return adjust_brightness(frames, 0.8)
    return frames

def process_frame_sequence(seq_dir, transform_type, output_dir):
    """Process a single frame sequence directory."""
    seq_dir = Path(seq_dir)
    frames = load_frame_sequence(seq_dir)
    
    transform_name = VALID_TRANSFORMATIONS[transform_type]
    transformed_dir = output_dir / seq_dir.name
    transformed_frames = transform_frames(frames, transform_type)
    save_frame_sequence(transformed_frames, transformed_dir)

def process_ride_directory(src_ride_dir, transform_type, heat_id, dest_rides_dir):
    """Process all frame sequences and create visualization videos in a ride directory."""
    src_ride_dir = Path(src_ride_dir)
    src_seqs_dir = src_ride_dir / 'seqs'
    
    if not src_seqs_dir.exists():
        print(f"Error: No sequences directory found in {src_ride_dir}")
        return
    
    # Create destination ride and sequence directories
    dest_ride_dir = dest_rides_dir / src_ride_dir.name
    dest_seqs_dir = dest_ride_dir / 'seqs'
    dest_seqs_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy sequence labels file
    src_labels = src_ride_dir / 'seq_labels.csv'
    if src_labels.exists():
        dest_labels = dest_ride_dir / 'seq_labels.csv'
        dest_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_labels, dest_labels)
    else:
        print(f"Warning: No seq_labels.csv found in {src_ride_dir}")
    
    # Process frame sequences (model input)
    print("Processing frame sequences...")
    for seq_dir in src_seqs_dir.glob('seq_*'):
        if seq_dir.is_dir():
            process_frame_sequence(seq_dir, transform_type, dest_seqs_dir)
    
    # Create visualization videos
    print("Creating visualization videos...")
    for video_file in src_ride_dir.glob('*.mp4'):
        # Skip already augmented videos
        if any(t in video_file.name for t in VALID_TRANSFORMATIONS.values()):
            continue
            
        # Extract ride number from the directory name since it matches the ride number
        ride_num = src_ride_dir.name
            
        frames = load_video(video_file)
        transform_name = VALID_TRANSFORMATIONS[transform_type]
        # Create augmented video with naming pattern that ScoreDataset expects: {heat_id}_{ride_num}.mp4
        output_path = dest_ride_dir / f"{heat_id}_{ride_num}.mp4"
        if not output_path.exists():
            transformed_frames = transform_frames(frames, transform_type)
            save_video(transformed_frames, output_path)

def process_heat_directory(heat_dir, transform_type=None):
    """Process all rides in a heat directory."""
    heat_dir = Path(heat_dir)
    src_rides_dir = heat_dir / 'rides'
    
    if not src_rides_dir.exists():
        print(f"Error: No rides directory found in {heat_dir}")
        return
    
    # Get heat ID from directory name
    heat_id = heat_dir.name
    
    # Determine which transformations to apply
    transforms_to_apply = []
    if transform_type:
        # Single transformation requested
        transform_name = VALID_TRANSFORMATIONS[transform_type]
        augmented_heat = heat_dir.parent / f"{heat_id}_{transform_name}"
        if not augmented_heat.exists():
            transforms_to_apply.append(transform_type)
        else:
            print(f"Skipping already-done transformation: {heat_id} ({transform_type})")
    else:
        # Check all possible transformations
        for t_type in VALID_TRANSFORMATIONS:
            transform_name = VALID_TRANSFORMATIONS[t_type]
            augmented_heat = heat_dir.parent / f"{heat_id}_{transform_name}"
            if not augmented_heat.exists():
                transforms_to_apply.append(t_type)
            else:
                print(f"Skipping already-done transformation: {heat_id} ({t_type})")
    
    if not transforms_to_apply:
        print(f"No new transformations to apply to heat: {heat_id}")
        return
        
    # Process each transformation
    for t_type in transforms_to_apply:
        transform_name = VALID_TRANSFORMATIONS[t_type]
        dest_heat_dir = heat_dir.parent / f"{heat_id}_{transform_name}"
        dest_rides_dir = dest_heat_dir / 'rides'
        dest_rides_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating transformed heat directory: {dest_heat_dir}")
        
        # Copy ride_times.csv for score prediction compatibility
        src_ride_times = heat_dir / 'ride_times.csv'
        if src_ride_times.exists():
            dest_ride_times = dest_heat_dir / 'ride_times.csv'
            shutil.copy2(src_ride_times, dest_ride_times)
        
        # Process each ride directory
        for src_ride_dir in src_rides_dir.glob('*'):
            if src_ride_dir.is_dir():
                print(f"Processing ride: {src_ride_dir.name}")
                # Use augmented heat_id for video naming: "123_flipped" instead of "123"
                augmented_heat_id = f"{heat_id}_{transform_name}"
                process_ride_directory(src_ride_dir, t_type, augmented_heat_id, dest_rides_dir)

def cleanup_augmented_heats():
    """Remove all augmented heat directories."""
    data_dir = Path(heats_path)
    removed_count = 0
    
    # Get all augmented heat directories (those with transformation suffixes)
    for heat_dir in data_dir.glob('*'):
        if heat_dir.is_dir() and any(t in heat_dir.name for t in VALID_TRANSFORMATIONS.values()):
            print(f"Removing augmented heat directory: {heat_dir}")
            shutil.rmtree(heat_dir)
            removed_count += 1
    
    print(f"\nCleanup complete. Removed {removed_count} augmented heat directories.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate augmented versions of surf maneuver videos.')
    parser.add_argument('--heat', type=str, help='Process only this specific heat ID')
    parser.add_argument('--transform', type=str, choices=list(VALID_TRANSFORMATIONS.keys()),
                      help='Apply only this specific transformation')
    parser.add_argument('--cleanup', action='store_true',
                      help='Remove all augmented heat directories')
    return parser.parse_args()

def main():
    """Main function to process all heat directories."""
    args = parse_args()
    
    # Handle cleanup operation
    if args.cleanup:
        cleanup_augmented_heats()
        return
        
    data_dir = Path(heats_path)
    
    # Build list of heats to process
    if args.heat:
        heat_dir = data_dir / args.heat
        if not heat_dir.is_dir():
            print(f"Error: Heat directory not found: {args.heat}")
            return
        heat_list = [heat_dir]
    else:
        heat_list = [d for d in data_dir.iterdir() if d.is_dir()]
    
    # Process each non-augmented heat
    for heat_dir in heat_list:
        if any(t in heat_dir.name for t in VALID_TRANSFORMATIONS.values()):
            print(f"Skipping transformed heat: {heat_dir.name}")
            continue
        print(f"Processing heat directory: {heat_dir}")
        process_heat_directory(heat_dir, args.transform)

if __name__ == '__main__':
    main() 