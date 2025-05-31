import torch
from torch.nn.utils.rnn import pad_sequence

def collate_variable_length_videos(batch):
    """
    Custom collate function for handling variable-length videos in DataLoader.
    
    Args:
        batch: List of (video_tensor, score) tuples from ScoreDataset
               video_tensor: [num_frames, channels, height, width] - variable num_frames
               score: scalar tensor
    
    Returns:
        videos: Batched videos [batch_size, max_frames, channels, height, width]
        scores: Batched scores [batch_size]
        attention_mask: Boolean mask [batch_size, max_frames] indicating real vs padded frames
    """
    
    videos = [item[0] for item in batch]  # List of variable-length video tensors
    scores = [item[1] for item in batch]  # List of score tensors
    
    # Use PyTorch's built-in pad_sequence (pads along first dimension by default)
    # pad_sequence returns [max_frames, batch_size, channels, height, width]
    padded_videos = pad_sequence(videos, batch_first=True, padding_value=0.0)
    
    # Create attention mask
    sequence_lengths = [video.shape[0] for video in videos]
    max_frames = padded_videos.shape[1]
    attention_mask = create_attention_mask(sequence_lengths, max_frames)
    
    # Stack scores into batch tensor
    batched_scores = torch.stack(scores)
    
    return padded_videos, batched_scores, attention_mask

def create_attention_mask(sequence_lengths, max_length):
    """
    Create attention mask for padded sequences.
    
    Args:
        sequence_lengths: List of actual sequence lengths
        max_length: Maximum sequence length (padding target)
    
    Returns:
        attention_mask: Boolean tensor [batch_size, max_length]
    """
    batch_size = len(sequence_lengths)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, length in enumerate(sequence_lengths):
        attention_mask[i, :length] = True
    
    return attention_mask 