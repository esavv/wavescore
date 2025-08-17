# This script runs inference on a surf video to detect maneuvers performed in the ride.
# It expects a video file as input and a trained model file in the ../models/ directory.

# Usage:
# src $ python inference.py --mode dev

import argparse, csv, gc, os, shutil, sys, torch
from checkpoints import load_checkpoint
from maneuver_model import SurfManeuverModel
from utils import (
    load_maneuver_taxonomy,
    iterate_video_sequences,
    set_device
)

# Set device to GPU if available, otherwise use CPU
device = set_device("inference")

model_dir = "../../models"
data_dir = "../../data"

def run_inference(video_path, model_filename, mode='dev'):
    try:
        # Load the video target for inference
        print('Loading target video...')
        video_dir = os.path.dirname(video_path)

        # Retrieve the model from locally. TODO: Support S3 retrieval
        print('Retrieving the model...')
        model_path = os.path.join(model_dir, model_filename)

        # Load the saved model
        print('Loading the model...')
        model = SurfManeuverModel(mode=mode)
        
        # Load checkpoint and handle both old and new formats
        model_state = load_checkpoint(model_path)[0]  # Only get the model state
        model.load_state_dict(model_state)
        
        model = model.to(device) 
               
        model.eval()
        
        # Run inference on each in-memory sequence (no disk IO)
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
        for sq, start_time, end_time, frames_tensor in iterate_video_sequences(
            video_path=video_path, sequence_duration=2, mode=mode
        ):
            # Run inference on the in-memory frames
            maneuver_id, confidence_scores = infer_sequence_tensor(model, frames_tensor, mode=mode)

            # lookup maneuver name
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

                indicator = ""
                if is_predicted and is_actual:
                    indicator = " <-- PREDICTED & ACTUAL"
                elif is_predicted:
                    indicator = " <-- PREDICTED"
                elif is_actual:
                    indicator = " <-- ACTUAL"

                print(f"    {class_id} ({class_name}): {score:.4f}{indicator}")

            confidence_data.append({
                'sequence': sq,
                'time_range': f"{start_time:.1f}s - {end_time:.1f}s",
                'predicted': maneuver_id,
                'actual': actual_labels.get(sq, None),
                'scores': {class_id: float(score) for class_id, score in enumerate(confidence_scores)}
            })

        # Return the results without activation analysis
        return maneuvers, confidence_data, taxonomy

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
    finally:
        # Explicitly free model and cached CUDA memory if applicable
        print("Maneuver inference: beginning cleanup...")
        try:
            del model
        except Exception:
            pass
        gc.collect()
        try:
            if device.type == 'cuda':
                print("Maneuver inference: clearing CUDA caches...")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

def infer_sequence_tensor(model, frames_tensor, mode='dev', hidden=None):
    """Run inference on a single sequence given a preprocessed frames tensor."""
    # Add batch dimension and move to device
    sequence = frames_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output, _ = model(sequence, None)
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
    models = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    if not models:
        print("Error: No model files found in " + model_dir)
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
    video_path = data_dir + "/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4"
    maneuvers, confidence_data, taxonomy = run_inference(video_path, model_filename, mode)
    print("\nPrediction dict: " + str(maneuvers))
