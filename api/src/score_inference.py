# This script runs inference on a surf video to predict the score of the ride.
# It expects a video file as input and a trained model file in the ../models/ directory.

# Usage:
# src $ python score_inference.py --mode dev

import argparse, gc, os, sys, torch
from checkpoints import load_checkpoint
from score_dataset import load_video_for_inference
from score_model import VideoScorePredictor
from utils import set_device

model_dir = "../../models"
data_dir = "../../data"

def run_inference(video_paths, model_filename, mode='dev'):
    try:
        # Set device to GPU if available, otherwise use CPU
        device = set_device("score_inference")

        # Retrieve the model from locally. TODO: Support S3 retrieval
        print('Retrieving the model...')
        model_path = os.path.join(model_dir, model_filename)

        # Load the saved model
        print('Loading the model...')
        model_state, _, _, _, training_config, _ = load_checkpoint(model_path)
        
        # Extract model configuration from training config
        model_type = training_config.get('model_type', 'clip')
        variant = training_config.get('variant', 'base')
        freeze_backbone = training_config.get('freeze_backbone', True)
        
        # Initialize model with same configuration as training
        model = VideoScorePredictor(
            model_type=model_type,
            variant=variant,
            freeze_backbone=freeze_backbone
        )
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        predicted_scores = []
        for video_path in video_paths:        
            # Load and preprocess the video
            print('Loading and preprocessing video...')
            video_tensor = load_video_for_inference(video_path, mode=mode)
            video_tensor = video_tensor.to(device)

            # Run inference on the video
            print('Running inference...')
            predicted_score = infer_video_score(model, video_tensor)
            predicted_scores.append(round(predicted_score, 2))
            
            # Load actual score if available
            actual_score = None
            score_path = os.path.join(os.path.dirname(video_path), 'score.csv')
            if os.path.exists(score_path):
                with open(score_path, 'r') as f:
                    next(f)  # Skip header
                    actual_score = float(next(f).strip())
            
            print(f"\n=== INFERENCE RESULTS ===")
            print(f"Video: {os.path.basename(video_path)}")
            print(f"Model: {model_type.upper()}-{variant}")
            if actual_score is not None:
                print(f"Predicted score: {predicted_score:.2f}/10.00 (Actual score: {actual_score:.2f})\n")
            else:
                print(f"Predicted score: {predicted_score:.2f}/10.00\n")
        return predicted_scores

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
    finally:
        # Explicitly free model and cached CUDA memory if applicable
        print("Score inference: beginning cleanup...")
        try:
            del model, video_tensor
        except Exception:
            pass
        gc.collect()
        try:
            if device.type == 'cuda':
                print("Score inference: clearing CUDA caches...")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

def infer_video_score(model, video_tensor):
    """Run inference on a single video."""
    with torch.no_grad():
        # Forward pass
        output = model(video_tensor)
        
        # Extract score (assuming model outputs a single value)
        predicted_score = output.squeeze().cpu().item()
        
        # Clamp score to valid range [0.0, 10.0]
        predicted_score = max(0.0, min(10.0, predicted_score))
        
    return predicted_score

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

    # Run inference on test videos (from training data)
    video_paths = [
        data_dir + "/inference_vids/1Zj_jAPToxI_6_inf/1Zj_jAPToxI_6_inf.mp4",
        data_dir + "/inference_vids/_Lwdbce6a4E_1_inf/_Lwdbce6a4E_1.mp4",
        data_dir + "/inference_vids/kl6bwSUqUw4_7_inf/kl6bwSUqUw4_7.mp4",
        data_dir + "/inference_vids/kl6bwSUqUw4_14_inf/kl6bwSUqUw4_14.mp4"
    ]
    run_inference(video_paths, model_filename, mode)
