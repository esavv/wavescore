# Surf Score Prediction Model

## Overview
This document outlines the plan for implementing a deep learning model to predict surf competition scores from raw video footage. The model will take a video of a surf ride as input and output a predicted score between 0.0 and 10.0.

## Architecture Approach

### Model Type
We will use a frame-based approach with temporal pooling:
- CLIP or ViT for frame-level feature extraction
- Temporal pooling (mean, max, or attention-based) for sequence-level features
- Simple MLP for final score prediction

This approach was chosen because it:
- Naturally handles variable-length videos without padding
- Leverages state-of-the-art image understanding models
- Is simpler to implement and debug
- Has proven effective in similar video understanding tasks
- Allows flexible temporal modeling through different pooling strategies

### Previous Approach (Not Used)
We initially considered video transformer architectures (TimeSformer, Video Swin) because they:
- Claimed to handle variable-length video inputs
- Had pre-trained weights available
- Were well-documented for transfer learning
- Had reasonable model sizes for local development
- Could capture both spatial and temporal features

However, we found that:
- These models don't actually support variable-length videos out of the box
- They require padding and attention masks, which complicates the implementation
- The attention mask mechanism isn't as straightforward as documented
- The models are more complex than needed for our use case

### Model Abstraction
We will implement a flexible architecture that allows easy experimentation with different pre-trained models:

```python
class VideoScorePredictor(nn.Module):
    def __init__(self, model_type='clip', variant='base'):
        # Initialize either CLIP or ViT
        # Add temporal pooling layer
        # Add regression head for score prediction
```

This abstraction allows us to:
- Switch between CLIP and ViT
- Experiment with different temporal pooling strategies
- Keep the interface consistent regardless of the underlying model

## Input Processing

### Variable Length Handling
- Sample frames at a fixed rate (e.g., 1 frame per second)
- Process each frame through image encoder
- Apply temporal pooling over frame embeddings
- No padding required - natural handling of variable lengths

### Frame Processing
- Frame resolution: 224x224 (standard for CLIP/ViT)
- Frame sampling rate: TBD
- Color channels: RGB
- Normalization: Model-specific preprocessing

## Score Prediction

### Output
- Single continuous value between 0.0 and 10.0
- Represents the average score from competition judges
- No need to handle discrete increments

### Loss Function
- Regression loss (MSE, MAE, or Huber)
- No need for ordinal regression or discrete classification

## Training Strategy

### Transfer Learning
- Start with pre-trained CLIP/ViT weights
- Fine-tune on surfing data
- Option to freeze encoder layers

### Regularization
- Dropout
- Weight decay
- Early stopping
- Cross-validation

## Score Labels

### Data Storage
Score labels will be stored in the existing `ride_times.csv` files within each heat directory:
- Location: `heats/{heat_id}/ride_times.csv`
- New column: `score` (float, range 0.0-10.0)
- Each row represents one ride with its corresponding score

### File Format
```csv
start_time,end_time,score
12.5,45.2,7.83
50.1,78.9,4.25
...
```

This approach keeps score data co-located with ride timing data for easy management.

## Implementation Plan Summary

### Completed Tasks
1. [COMPLETED] Set up training infrastructure (`score_train.py`)
2. [COMPLETED] Create inference pipeline (`score_inference.py`)
3. [COMPLETED] Set up model abstraction (`score_model.py`)
4. [COMPLETED] Implement data loading pipeline (`score_dataset.py`)
5. [COMPLETED] Implement variable-length video batching support (`collate.py`)
6. [COMPLETED] Update data augmentation for score prediction compatibility (`augment_data.py`)

### Pivot to Frame-Based Approach
After implementing the initial video transformer approach, we discovered limitations with variable-length video handling. We are pivoting to a frame-based approach using CLIP/ViT with temporal pooling.

### New Tasks
7. [COMPLETED] Implement new frame-based architecture (`score_model.py`)
8. [COMPLETED] Adapt dataset for frame-based approach (`score_dataset.py`)
9. [COMPLETED] Adapt training pipeline for frame-based model (`score_train.py`)
10. [COMPLETED] Adapt inference pipeline for frame-based model (`score_inference.py`)
11. [COMPLETED] Clean up deprecated code

## Implementation Plan Details

### 1. Set up training infrastructure (`score_train.py`)
**Key Functions:**
- `train_epoch()` - single epoch training loop
- `validate_epoch()` - validation loop with metrics
- `main_training_loop()` - full training process
- `configure_optimizer()` - optimizer and scheduler setup
- `calculate_metrics()` - MAE, MSE, R² score calculation
- `save_training_progress()` - logging and checkpoint saving

**Usage:**
- Run from command line: `python score_train.py --mode [dev|prod]`
- Script prompts user to:
  - Choose between training new model or resuming from checkpoint
  - Select model type (TimeSformer or Video Swin)
- Training progress logged to console and saved checkpoints
- Dev mode uses smaller dataset for faster iteration
- Prod mode uses full dataset for final training

**Implementation Notes:**
- Start with skeleton of training script following pattern of existing train.py
- Define clear interfaces needed from model and dataset classes
- Implement training loop structure before model/dataset implementation
- Add placeholder calls to model and dataset that will be implemented later

### 2. Create inference pipeline (`score_inference.py`)
**Key Functions:**
- `ScorePredictor` class - simplified inference wrapper
- `predict_single_video()` - score prediction for one video
- `batch_predict()` - process multiple videos
- `load_trained_model()` - load model from checkpoint
- `visualize_prediction()` - show video with predicted score

**Usage:**
- Run from command line: `python score_inference.py`
- Script prompts user to:
  - Select which model checkpoint to use for inference
- Video path for inference is hard-coded in the script
- Output: predicted score and optional visualization

**Implementation Notes:**
- Follow similar patterns to training script
- Use same model and dataset interfaces as training
- Ensure consistent checkpoint handling
- Add placeholder calls to model and dataset that will be implemented later

### 3. Set up model abstraction (`score_model.py`)
**Key Functions:**
- `VideoScorePredictor` class - main model wrapper
- `load_timesformer()` - initialize TimeSformer with pre-trained weights
- `load_video_swin()` - initialize Video Swin Transformer with pre-trained weights
- `create_regression_head()` - add score prediction layer
- `freeze_backbone_layers()` - optional layer freezing for transfer learning

**Usage:**
- Imported by `score_train.py` and `score_inference.py`
- Example usage in training script:
```python
from score_model import VideoScorePredictor

# Initialize model
model = VideoScorePredictor(model_type='timesformer', variant='base')
model.load_timesformer()  # Load pre-trained weights
model.create_regression_head()  # Add score prediction layer
```

### 4. Implement data loading pipeline (`score_dataset.py`)
**Key Functions:**
- `ScoreDataset` class - PyTorch dataset for video-score pairs
- `sample_frames()` - extract frames at fixed rate from videos
- `preprocess_frames()` - resize, normalize, convert to tensor
- `load_score_labels()` - load scores from ride_times.csv files
- `get_video_score_pairs()` - map videos to their scores
- `parse_ride_times_with_scores()` - read CSV with new score column

**Usage:**
- Imported by `score_train.py` and `score_inference.py`
- Example usage in training script:
```python
from score_dataset import ScoreDataset

# Create dataset pointing to root directory of all heats
dataset = ScoreDataset(
    root_dir="../data/heats",  # Root directory containing all heat directories
    transform=None,  # Optional video transforms
    mode='train'  # 'train', 'val', or 'test'
)
```

The dataset will:
1. Recursively find all heat directories under root_dir
2. For each heat directory, read the ride_times.csv file to get scores
3. Map each video to its corresponding score
4. Handle train/val/test splits internally

### 5. Implement variable-length video batching support (`collate.py`)
**Key Functions:**
- `collate_variable_length_videos()` - custom collate function for DataLoader to handle variable-length videos
- `pad_sequence_batch()` - pad videos in a batch to the same length
- `create_attention_mask()` - create attention masks for padded sequences

**Purpose:**
The current dataset implementation uses sample rates instead of fixed frame counts, resulting in variable-length video tensors (e.g., 15 frames vs 50 frames). PyTorch's default DataLoader cannot batch these together because tensor dimensions must match for stacking.

**Technical Challenge:**
- Video A: `[15, 3, 224, 224]` (short ride)
- Video B: `[50, 3, 224, 224]` (long ride)
- Default batching fails: cannot stack tensors with different first dimensions

**Solution Approach:**
1. **Custom Collate Function**: Implement a function that receives a list of variable-length videos and scores
2. **Dynamic Padding**: Pad shorter videos to match the longest video in each batch
3. **Attention Masks**: Create masks to tell TimeSformer which frames are real vs padded
4. **Efficient Batching**: Only pad to the maximum length within each batch (not globally)
5. **Update score_train.py**: Modify DataLoader creation to conditionally use collate function when batch_size > 1

**Usage:**
- Imported by `score_train.py` and integrated with DataLoader when batch_size > 1
- Example usage in training script:
```python
from collate import collate_variable_length_videos

# Use collate function only when batch_size > 1
if batch_size > 1:
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_variable_length_videos
    )
else:
    # batch_size = 1 works fine with default collating
    dataloader = DataLoader(dataset, batch_size=1)
```

**Integration with score_train.py:**
- Update DataLoader creation logic to conditionally use collate function
- Add import statement for collate_variable_length_videos
- Apply collate function when batch_size > 1 (required for variable-length videos)
- Use default collating when batch_size = 1 (no batching conflicts)

**Benefits:**
- Enables batch training with batch_size > 1
- Preserves variable-length advantages of transformer architecture
- More efficient than forcing batch_size=1
- Maintains temporal fidelity of original videos

**Implementation Notes:**
- TimeSformer supports attention masks for handling padded sequences
- Padding should use zeros or repeat last frame
- Masks should be boolean tensors indicating valid vs padded positions
- Function should handle both videos and scores in batch format

### 6. Update data augmentation for score prediction compatibility (`augment_data.py`)
**Key Functions:**
- `copy_score_data()` - copy ride_times.csv and ride videos to augmented heat directories
- `augment_heat_for_score_prediction()` - extend existing augmentation to handle score prediction files
- `validate_augmented_score_data()` - verify augmented heats work with ScoreDataset

**Purpose:**
The existing `augment_data.py` handles maneuver prediction data augmentation but needs updates to support score prediction. Score prediction requires `ride_times.csv` files and individual ride videos (`{heat_id}_{ride_idx}.mp4`) to be copied to augmented heat directories.

**Current Gap:**
- `augment_data.py` currently augments frame sequences for maneuver prediction
- Score prediction needs: `ride_times.csv` + individual ride videos
- Augmented heats must be compatible with `ScoreDataset` expectations

**Solution Approach:**
1. **Extend heat augmentation**: When creating augmented heat directories, copy score prediction files
2. **Copy ride_times.csv**: Ensure score labels are preserved in augmented heats
3. **Copy ride videos**: Copy `{heat_id}_{ride_idx}.mp4` files to augmented heat structure
4. **Maintain compatibility**: Augmented heats should work seamlessly with `ScoreDataset`

**Implementation Notes:**
- Update existing `augment_data()` function to handle score prediction files
- Add logic to copy `ride_times.csv` from original to augmented heat directories
- Add logic to copy ride videos from `rides/{idx}/{heat_id}_{idx}.mp4` structure
- Ensure augmented heat naming conventions work with ScoreDataset path expectations
- Test that `ScoreDataset` can load augmented heats without modification

### 7. Implement new frame-based architecture (`score_model.py`)
**Key Functions:**
- `VideoScorePredictor` class - main model wrapper
- `_initialize_encoder()` - initialize CLIP or ViT
- `_create_temporal_pooling()` - create pooling layer (attention, mean, or max)
- `_create_regression_head()` - add score prediction layer
- `forward()` - process frames through encoder, pooling, and head

**Implementation Notes:**
- Use CLIP or ViT from Hugging Face transformers
- Implement temporal pooling strategies within the model:
  - Attention-based pooling (default)
  - Mean pooling (simpler alternative)
  - Max pooling (alternative)
- Keep regression head similar to current implementation
- Remove all padding/attention mask logic

### 8. Adapt dataset for frame-based approach (`score_dataset.py`)
**Changes Required:**
- Remove padding and attention mask logic
- Keep frame sampling and preprocessing
- Update `__getitem__` to return only frames and score
- Remove `_pad_video` method
- Update frame preprocessing for CLIP/ViT requirements

### 9. Adapt training pipeline for frame-based model (`score_train.py`)
**Changes Required:**
- Remove custom collate function
- Update DataLoader creation
- Adapt training loop for frame-based model
- Keep checkpoint and logging functionality

### 10. Adapt inference pipeline for frame-based model (`score_inference.py`)
**Changes Required:**
- Remove attention mask handling
- Update model loading for new architecture
- Keep score clamping and output formatting

### 11. Clean up deprecated code
**Files to Remove:**
- `collate.py` - no longer needed
- Remove padding/attention mask code from other files
