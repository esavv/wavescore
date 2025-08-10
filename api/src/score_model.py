import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, ViTModel

class VideoScorePredictor(nn.Module):
    def __init__(self, model_type='clip', variant='base', freeze_backbone=True, dropout_rate=0.5, pooling_type='attention', encode_chunk_size: int = 32):
        super(VideoScorePredictor, self).__init__()
        self.model_type = model_type
        self.variant = variant
        self.freeze_backbone = freeze_backbone
        self.pooling_type = pooling_type
        self.encode_chunk_size = max(1, int(encode_chunk_size))
        
        # Initialize the image encoder
        self._initialize_encoder()
        
        # Create temporal pooling layer
        self._create_temporal_pooling()
        
        # Create regression head
        self._create_regression_head(dropout_rate)
        
        # Apply backbone freezing if specified
        if freeze_backbone:
            self._freeze_backbone_layers()
    
    def _initialize_encoder(self):
        """Initialize CLIP or ViT encoder with pre-trained weights."""
        print(f"Loading {self.model_type.upper()}-{self.variant} encoder...")
        
        if self.model_type == 'clip':
            if self.variant == 'base':
                model_name = "openai/clip-vit-base-patch32"
                self.hidden_size = 768
            elif self.variant == 'large':
                model_name = "openai/clip-vit-large-patch14"
                self.hidden_size = 1024
            else:
                raise ValueError(f"Unsupported CLIP variant: {self.variant}")
            
            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            
        elif self.model_type == 'vit':
            if self.variant == 'base':
                model_name = "google/vit-base-patch16-224"
                self.hidden_size = 768
            elif self.variant == 'large':
                model_name = "google/vit-large-patch16-224"
                self.hidden_size = 1024
            else:
                raise ValueError(f"Unsupported ViT variant: {self.variant}")
            
            self.encoder = ViTModel.from_pretrained(model_name)
            
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose 'clip' or 'vit'")
        
        print(f"  ✓ {self.model_type.upper()} encoder loaded successfully")
    
    def _create_temporal_pooling(self):
        """Create temporal pooling layer for aggregating frame embeddings."""
        print(f"Creating {self.pooling_type} temporal pooling layer...")
        
        if self.pooling_type == 'attention':
            # Attention-based pooling
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 2, 1)
            )
        elif self.pooling_type in ['mean', 'max']:
            # No learnable parameters for mean/max pooling
            pass
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}. Choose 'attention', 'mean', or 'max'")
        
        print(f"  ✓ {self.pooling_type} temporal pooling layer created successfully")
    
    def _create_regression_head(self, dropout_rate):
        """Create regression head for score prediction."""
        print("Creating regression head for score prediction...")
        
        # Regression head: hidden_size -> 1 (score between 0.0 and 10.0)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Sigmoid to ensure output is between 0 and 1, then scale to 0-10
        )
        
        print("  ✓ Regression head created successfully")
    
    def _freeze_backbone_layers(self):
        """Freeze encoder layers for transfer learning."""
        print("Freezing encoder layers...")
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        print("  ✓ Encoder layers frozen successfully")
    
    def _apply_temporal_pooling(self, frame_embeddings):
        """Apply the selected temporal pooling strategy to frame embeddings."""
        if self.pooling_type == 'attention':
            # Reshape to [batch_size * num_frames, hidden_size] for linear layer
            batch_size, num_frames, hidden_size = frame_embeddings.shape
            reshaped_embeddings = frame_embeddings.view(-1, hidden_size)
            
            # Apply attention
            attention_weights = self.temporal_attention(reshaped_embeddings)  # [batch_size * num_frames, 1]
            attention_weights = attention_weights.view(batch_size, num_frames, 1)  # [batch_size, num_frames, 1]
            attention_weights = F.softmax(attention_weights, dim=1)  # Normalize weights
            
            # Weighted sum
            pooled_embeddings = torch.sum(frame_embeddings * attention_weights, dim=1)  # [batch_size, hidden_size]
            
        elif self.pooling_type == 'mean':
            # Mean pooling
            pooled_embeddings = frame_embeddings.mean(dim=1)  # [batch_size, hidden_size]
        elif self.pooling_type == 'max':
            # Max pooling
            pooled_embeddings = frame_embeddings.max(dim=1)[0]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")
        
        return pooled_embeddings
    
    def forward(self, frames):
        """
        Forward pass for score prediction.
        
        Args:
            frames: Input tensor of shape [batch_size, num_frames, channels, height, width]
            
        Returns:
            scores: Predicted scores tensor of shape [batch_size] with values in range [0.0, 10.0]
        """
        batch_size, num_frames = frames.shape[:2]
        hidden_size = self.hidden_size

        # Encode frames in temporal chunks to reduce peak memory
        embeddings_per_chunk = []
        chunk_size = self.encode_chunk_size
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            # [B, chunk, C, H, W] -> [B*chunk, C, H, W]
            chunk = frames[:, start:end].contiguous().view(-1, *frames.shape[2:])
            encoder_output = self.encoder(chunk)
            chunk_embed = (
                encoder_output.pooler_output
                if hasattr(encoder_output, 'pooler_output') and encoder_output.pooler_output is not None
                else encoder_output.last_hidden_state[:, 0, :]
            )  # [B*chunk, hidden]
            # [B*chunk, hidden] -> [B, chunk, hidden]
            chunk_frames = end - start
            chunk_embed = chunk_embed.view(batch_size, chunk_frames, hidden_size)
            embeddings_per_chunk.append(chunk_embed)

        # Concatenate across chunks to recover full [B, T, hidden]
        frame_embeddings = torch.cat(embeddings_per_chunk, dim=1)
        
        # Apply temporal pooling
        pooled_embeddings = self._apply_temporal_pooling(frame_embeddings)
        
        # Forward pass through regression head
        scores = self.regression_head(pooled_embeddings)
        
        # Scale scores from [0, 1] to [0, 10]
        scores = scores * 10.0
        
        return scores.squeeze(-1)  # Remove last dimension to get [batch_size] 