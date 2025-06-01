import os
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class ScoreDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', dev_mode='dev', model_type='clip'):
        """
        Dataset for loading surf videos with their corresponding scores.
        
        Args:
            root_dir: Root directory containing heat directories (e.g., "../data/heats")
            transform: Optional video transforms
            mode: 'train', 'val', or 'test' for data splitting
            dev_mode: 'dev' or 'prod' for development vs production settings
            model_type: 'clip' or 'vit' to specify model-specific processing
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.dev_mode = dev_mode
        self.model_type = model_type
        
        # Video processing settings based on dev_mode
        if dev_mode == 'dev':
            self.frame_size = 224  # Standard size for CLIP/ViT
            self.sample_rate = 0.10  # Keep 10% of frames (every 10th frame) for faster processing
        else:  # prod mode
            self.frame_size = 224  # Standard size for CLIP/ViT
            self.sample_rate = 0.33  # Keep 33% of frames (every 3rd frame) for better temporal resolution
        
        # Load all video-score pairs
        self.samples = self._load_video_score_pairs()
        
        print(f"> Loaded {len(self.samples)} video-score pairs (full dataset)")
    
    def _load_video_score_pairs(self):
        """Load all video-score pairs from the heats directory."""
        samples = []
        
        print(f"> Loading video-score pairs from {self.root_dir}")
        
        # Iterate through each heat directory
        for heat_dir in os.listdir(self.root_dir):
            heat_path = os.path.join(self.root_dir, heat_dir)
            if not os.path.isdir(heat_path):
                continue
            
            heat_id = heat_dir
            ride_times_path = os.path.join(heat_path, "ride_times.csv")
            rides_dir = os.path.join(heat_path, "rides")
            
            # Skip if ride_times.csv doesn't exist
            if not os.path.exists(ride_times_path):
                print(f"  Warning: No ride_times.csv found in {heat_path}")
                continue
            
            # Load score labels from ride_times.csv
            try:
                scores_df = self._parse_ride_times_with_scores(ride_times_path)
                if scores_df is None:
                    print(f"  Warning: No scores found in {ride_times_path}")
                    continue
            except Exception as e:
                print(f"  Error loading scores from {ride_times_path}: {e}")
                continue
            
            # Match each ride video with its score
            heat_samples = 0
            for ride_idx, (_, row) in enumerate(scores_df.iterrows()):
                video_path = os.path.join(rides_dir, str(ride_idx), f"{heat_id}_{ride_idx}.mp4")
                
                if os.path.exists(video_path):
                    samples.append({
                        'video_path': video_path,
                        'score': row['score'],
                        'heat_id': heat_id,
                        'ride_idx': ride_idx
                    })
                    heat_samples += 1
                else:
                    print(f"  Warning: Video not found: {video_path}")
            
            print(f"  Heat '{heat_id}': {heat_samples} video-score pairs")
        
        return samples
    
    def _parse_ride_times_with_scores(self, csv_path):
        """Parse ride_times.csv file and extract scores from the third column."""
        try:
            df = pd.read_csv(csv_path)
            
            # Check if score column exists
            if 'score' not in df.columns:
                # For now, return None if no scores are present
                # In the future, scores will be added to the CSV files
                return None
            
            # Validate that we have the expected columns
            required_columns = ['start', 'end', 'score']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}, Found: {list(df.columns)}")
            
            # Convert score column to float and validate range
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            
            # Check for invalid scores
            invalid_scores = df[(df['score'] < 0) | (df['score'] > 10) | df['score'].isna()]
            if len(invalid_scores) > 0:
                print(f"  Warning: Found {len(invalid_scores)} invalid scores (must be 0.0-10.0)")
                df = df.dropna(subset=['score'])
                df = df[(df['score'] >= 0) & (df['score'] <= 10)]
            
            return df
            
        except Exception as e:
            print(f"  Error parsing {csv_path}: {e}")
            return None
    
    def _split_data(self, samples, mode):
        """Split data into train/val/test sets."""
        if mode not in ['train', 'val', 'test']:
            return samples
        
        # Simple split: 70% train, 20% val, 10% test
        # Use deterministic splitting based on hash of video path for consistency
        train_samples = []
        val_samples = []
        test_samples = []
        
        for sample in samples:
            # Use hash of video path for consistent splitting
            hash_val = hash(sample['video_path']) % 10
            if hash_val < 7:  # 70%
                train_samples.append(sample)
            elif hash_val < 9:  # 20%
                val_samples.append(sample)
            else:  # 10%
                test_samples.append(sample)
        
        if mode == 'train':
            return train_samples
        elif mode == 'val':
            return val_samples
        else:  # test
            return test_samples
    
    def _sample_frames_from_video(self, video_path):
        """Extract frames at specified sample rate, preserving variable video lengths."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0 or fps == 0:
            cap.release()
            raise ValueError(f"Invalid video properties: {video_path}")
        
        # Calculate sampling interval based on sample rate
        # sample_rate = 0.10 means keep every 10th frame
        # sample_rate = 0.33 means keep every 3rd frame  
        frame_interval = max(1, int(1.0 / self.sample_rate))
        
        # Sample frames at the calculated interval
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Keep frames based on sample rate
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        return frames
    
    def _preprocess_frames(self, frames):
        """Preprocess frames: resize, normalize, convert to tensor."""
        processed_frames = []
        
        for frame in frames:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Resize while preserving aspect ratio
            processed_frame = self._preserve_aspect_ratio_resize(pil_image, self.frame_size)
            
            processed_frames.append(processed_frame)
        
        # Stack frames: shape [num_frames, channels, height, width]
        video_tensor = torch.stack(processed_frames)
        
        return video_tensor
    
    def _preserve_aspect_ratio_resize(self, image, target_size):
        """Resize image while preserving aspect ratio, then pad to square."""
        width, height = image.size
        
        # Calculate scaling factor
        scale_factor = target_size / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create square background
        new_image = Image.new('RGB', (target_size, target_size), color=(0, 0, 0))
        
        # Center the resized image
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        new_image.paste(resized_image, (paste_x, paste_y))
        
        # Convert to tensor and normalize
        tensor_image = F.to_tensor(new_image)
        
        return tensor_image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a video-score pair."""
        sample = self.samples[idx]
        
        # Sample frames from video
        frames = self._sample_frames_from_video(sample['video_path'])
        
        # Preprocess frames
        video_tensor = self._preprocess_frames(frames)
        
        # Convert score to float32 tensor
        score = torch.tensor(sample['score'], dtype=torch.float32)
        
        return video_tensor, score

def load_video_for_inference(video_path, mode='dev'):
    """
    Load and preprocess a single video for inference.
    
    Args:
        video_path: Path to the video file
        mode: 'dev' or 'prod' for processing settings
        
    Returns:
        torch.Tensor: Preprocessed video tensor [1, num_frames, channels, height, width]
    """
    # Create a temporary dataset instance for preprocessing
    temp_dataset = ScoreDataset(
        root_dir="", 
        dev_mode=mode
    )
    
    # Load and preprocess the video
    frames = temp_dataset._sample_frames_from_video(video_path)
    video_tensor = temp_dataset._preprocess_frames(frames)
    
    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor 