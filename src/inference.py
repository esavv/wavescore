# This script runs inference on a surf video to detect maneuvers performed in the ride.
# It expects a video file as input and a trained model file in the ../models/ directory.

# Usage:
# src $ python inference.py --mode dev

import argparse, csv, os, shutil, sys, torch
from model import SurfManeuverModel
from utils import sequence_video_frames, load_frames_from_sequence, load_maneuver_taxonomy
from checkpoints import load_checkpoint

# Set device to GPU if available, otherwise use CPU
print('inference: Configuring device & model transforms...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def verify_input_sequence(sequence, seq_name, mode):
    """Print statistics about input sequence to verify loading."""
    print(f"\n=== INPUT SEQUENCE DIAGNOSTICS: {seq_name} ===")
    print(f"Input sequence shape: {sequence.shape}")
    print(f"Input sequence dtype: {sequence.dtype}")
    print(f"Input mode: {mode}")
    
    # Check sequence values
    min_val = sequence.min().item()
    max_val = sequence.max().item()
    mean_val = sequence.mean().item()
    std_val = sequence.std().item()
    
    print(f"Input sequence range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"Input sequence mean: {mean_val:.6f}, std: {std_val:.6f}")
    
    # Check if frames have variation
    if sequence.shape[1] > 1:  # More than one frame
        frame_diffs = []
        for i in range(1, min(5, sequence.shape[1])):  # Check first few frames
            diff = (sequence[0, i] - sequence[0, 0]).abs().mean().item()
            frame_diffs.append(diff)
            print(f"Difference between frame 0 and frame {i}: {diff:.6f}")
        
        if all(diff < 1e-6 for diff in frame_diffs):
            print("WARNING: First frames appear to be identical or very similar!")
    
    # Check if all values are zeros or very close to zero
    if max_val < 1e-6:
        print("WARNING: Sequence appears to be all zeros or very close to zeros!")
    
    # Sample stats for a few individual frames
    num_samples = min(3, sequence.shape[1])
    for i in range(num_samples):
        frame = sequence[0, i]
        print(f"Frame {i} stats - Min: {frame.min().item():.6f}, Max: {frame.max().item():.6f}, Mean: {frame.mean().item():.6f}")

def run_inference(video_path, model_filename, mode='dev'):
    # Load the video target for inference
    print('Loading target video...')
    video_dir = os.path.dirname(video_path)

    # If not already there, download the model from S3 to local directory
    # Model URL: 
    print('Retrieving the model...')
    model_dir = "../models/"
    model_path = os.path.join(model_dir, model_filename)

    # Load the saved model
    print('Loading the model...')
    model = SurfManeuverModel(mode=mode)
    
    # Load checkpoint and handle both old and new formats
    model_state = load_checkpoint(model_path)[0]  # Only get the model state
    model.load_state_dict(model_state)
    
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
    taxonomy = load_maneuver_taxonomy()

    # Load the sequence labels if they exist
    seq_labels_path = os.path.join(video_dir, 'seq_labels.csv')
    actual_labels = {}
    if os.path.exists(seq_labels_path):
        with open(seq_labels_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                seq_num = int(row['sequence_id'].split('_')[1])  # Extract number from 'seq_X'
                actual_labels[seq_num] = int(row['label'])

    maneuvers = []
    confidence_data = []
    sequence_duration = sequences_metadata['sequence_duration']
    total_sequences = sequences_metadata['total_sequences']
    
    # No need to initialize hidden state for 3D CNN
    
    for sq in range(total_sequences):
        seq_name = f'seq_{sq}'
        seq_dir = os.path.join(seqs_path, seq_name)
        if os.path.isdir(seq_dir):
            # Run inference on each sequence independently
            maneuver_id, confidence_scores = infer_sequence(model, seq_dir, mode=mode)
            
            start_time = sq * sequence_duration
            end_time = start_time + sequence_duration
            
            # lookup manuever name
            name = taxonomy.get(maneuver_id, 'Unknown maneuver')
            if maneuver_id != 0:
                maneuvers.append({'name': name, 'start_time': start_time, 'end_time': end_time})

            # Print the confidence scores for all classes
            print(f"\nSequence {sq} (Time: {start_time:.1f}s - {end_time:.1f}s):")
            print(f"  Predicted maneuver: {maneuver_id} ({name})")
            print("  Confidence scores:")
            for class_id, score in enumerate(confidence_scores):
                class_name = taxonomy.get(class_id, 'Unknown')
                actual_label = actual_labels.get(sq, None)
                is_predicted = class_id == maneuver_id
                is_actual = actual_label is not None and class_id == actual_label
                
                # Build the indicator string
                indicator = ""
                if is_predicted and is_actual:
                    indicator = " <-- PREDICTED & ACTUAL"
                elif is_predicted:
                    indicator = " <-- PREDICTED"
                elif is_actual:
                    indicator = " <-- ACTUAL"
                
                print(f"    {class_id} ({class_name}): {score:.4f}{indicator}")
            
            # Store confidence data for potential visualization
            confidence_data.append({
                'sequence': sq,
                'time_range': f"{start_time:.1f}s - {end_time:.1f}s",
                'predicted': maneuver_id,
                'actual': actual_labels.get(sq, None),
                'scores': {class_id: float(score) for class_id, score in enumerate(confidence_scores)}
            })

    # clean things up
    shutil.rmtree(seqs_path)

    # Return the results without activation analysis
    return maneuvers, confidence_data, taxonomy

def infer_sequence(model, seq_dir, mode='dev', hidden=None):
    """Run inference on a single sequence."""
    # Use the shared function from utils.py with batch dimension already added
    sequence = load_frames_from_sequence(seq_dir, transform=None, mode=mode, add_batch_dim=True)
    sequence = sequence.to(device)
    
    with torch.no_grad():
        # Forward pass (hidden parameter is ignored in the 3D CNN model)
        output, _ = model(sequence, None)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence_scores = probabilities.squeeze().cpu().numpy()
        predicted_class = output.argmax(dim=1).item()
        
    return predicted_class, confidence_scores

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
    maneuvers, confidence_data, taxonomy = run_inference(video_path, model_filename, mode)
    print("\nPrediction dict: " + str(maneuvers))
