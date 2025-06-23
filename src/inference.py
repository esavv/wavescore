# This script runs inference on a surf video to detect maneuvers performed in the ride.
# It expects a video file as input and a trained model file in the ../models/ directory.

# Usage:
# src $ python inference.py --mode dev

import argparse, csv, os, shutil, sys, torch
from checkpoints import load_checkpoint
from model import SurfManeuverModel
from utils import (
    load_frames_from_sequence, 
    load_maneuver_taxonomy, 
    sequence_video_frames, 
    set_device
)

# Set device to GPU if available, otherwise use CPU
device = set_device()

def run_inference(video_path, model_filename, mode='dev'):
    try:
        # Load the video target for inference
        print('Loading target video...')
        video_dir = os.path.dirname(video_path)

        # Retrieve the model from locally. TODO: Support S3 retrieval
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

    except Exception as e:
        # Clean up any temporary files
        if 'seqs_path' in locals() and os.path.exists(seqs_path):
            shutil.rmtree(seqs_path)
        print(f"Error during inference: {str(e)}")
        raise

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
