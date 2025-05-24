# Surf Score Prediction Model

## Overview
This document outlines the plan for implementing a deep learning model to predict surf competition scores from raw video footage. The model will take a video of a surf ride as input and output a predicted score between 0.0 and 10.0.

## Architecture Approach

### Model Type
We will use a video transformer architecture, specifically focusing on two pre-trained models:
- TimeSformer
- Video Swin Transformer

These models were chosen because they:
- Can handle variable-length video inputs naturally
- Have pre-trained weights available
- Are well-documented for transfer learning
- Have reasonable model sizes for local development
- Can capture both spatial and temporal features effectively

### Model Abstraction
We will implement a flexible architecture that allows easy experimentation with different pre-trained models:

```python
class VideoScorePredictor(nn.Module):
    def __init__(self, model_type='timesformer', variant='base'):
        # Initialize either TimeSformer or Video Swin Transformer
        # Add regression head for score prediction
```

This abstraction allows us to:
- Switch between TimeSformer and Video Swin Transformer
- Experiment with different model variants (base/large)
- Keep the interface consistent regardless of the underlying model

## Input Processing

### Variable Length Handling
- No padding required due to transformer architecture
- Sample frames at a fixed rate (e.g., 1 frame per second)
- Feed exact number of frames to model
- Attention mechanism handles variable lengths naturally

### Frame Processing
- Frame resolution: TBD (likely 224x224 or 384x384)
- Frame sampling rate: TBD (to be determined based on experimentation)
- Color channels: RGB

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
- Start with pre-trained weights
- Fine-tune on surfing data
- Option to freeze early layers

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

## Implementation Plan

**Note**: For each new file, we will include example command-line usage documentation to facilitate testing and experimentation during development.

### 1. Set up model abstraction (`score_model.py`)
**Key Functions:**
- `VideoScorePredictor` class - main model wrapper
- `load_timesformer()` - initialize TimeSformer with pre-trained weights
- `load_video_swin()` - initialize Video Swin Transformer with pre-trained weights
- `create_regression_head()` - add score prediction layer
- `freeze_backbone_layers()` - optional layer freezing for transfer learning

### 2. Implement data loading pipeline (`score_dataset.py`)
**Key Functions:**
- `ScoreDataset` class - PyTorch dataset for video-score pairs
- `sample_frames()` - extract frames at fixed rate from videos
- `preprocess_frames()` - resize, normalize, convert to tensor
- `load_score_labels()` - load scores from ride_times.csv files
- `get_video_score_pairs()` - map videos to their scores
- `parse_ride_times_with_scores()` - read CSV with new score column

### 3. Configure frame sampling and preprocessing
**Key Functions (in `score_dataset.py`):**
- `extract_frames_at_rate()` - sample frames at specified FPS
- `handle_variable_length()` - manage different video durations
- `apply_transforms()` - data augmentation transforms
- `create_data_loaders()` - train/val/test split and DataLoader creation

### 4. Set up training infrastructure (`score_train.py`)
**Key Functions:**
- `train_epoch()` - single epoch training loop
- `validate_epoch()` - validation loop with metrics
- `main_training_loop()` - full training process
- `configure_optimizer()` - optimizer and scheduler setup
- `calculate_metrics()` - MAE, MSE, R² score calculation
- `save_training_progress()` - logging and checkpoint saving

### 5. Extend shared infrastructure (shared files)
**Updates to existing files:**
- `checkpoints.py` - add score model checkpoint functions
- `model_logging.py` - add score training log format
- `utils.py` - add score-specific utility functions

### 6. Create inference pipeline (`score_inference.py`)
**Key Functions:**
- `ScorePredictor` class - simplified inference wrapper
- `predict_single_video()` - score prediction for one video
- `batch_predict()` - process multiple videos
- `load_trained_model()` - load model from checkpoint
- `visualize_prediction()` - show video with predicted score

### 7. Implement evaluation metrics and visualization (`score_evaluation.py`)
**Key Functions:**
- `evaluate_model()` - comprehensive model evaluation
- `plot_score_predictions()` - scatter plot of predicted vs actual scores
- `analyze_score_distribution()` - distribution analysis
- `calculate_judge_agreement()` - compare model to human variability

## File Structure
```
src/
├── score_model.py          # Score model architecture
├── score_dataset.py        # Score data loading and preprocessing
├── score_train.py          # Score training loop and evaluation
├── score_inference.py      # Score inference pipeline
├── score_evaluation.py     # Score evaluation and visualization
├── checkpoints.py          # Shared checkpoint management (EXTENDED)
├── model_logging.py        # Shared training logs (EXTENDED)
└── utils.py                # Shared utilities (EXTENDED)
```

## Shared Infrastructure Extensions

### `checkpoints.py`
- Add score model save/load functions
- Maintain backward compatibility with maneuver model checkpoints
- Support different model types in same interface

### `model_logging.py`
- Add score training log format
- Support regression metrics (MAE, MSE, R²)
- Maintain consistent logging structure

### `utils.py`
- Add score validation functions
- Add score formatting utilities
- Add score dataset statistics functions

## Dependencies to Add
- `timm` (for TimeSformer)
- `transformers` (for Video Swin Transformer)
- `decord` or `av` (for efficient video loading)
- Additional video processing libraries as needed

## Future Considerations

### Multi-task Learning
- Potential to combine with maneuver detection
- Shared backbone for both tasks
- Separate heads for maneuvers and scoring

### Data Augmentation
- Temporal augmentation
- Spatial augmentation
- Score-aware augmentation

### Model Evaluation
- Cross-validation strategy
- Comparison with human judges
- Error analysis
- Visualization of model attention 