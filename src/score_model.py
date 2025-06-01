import torch.nn as nn
from transformers import TimesformerModel
from transformers import VideoMAEModel

class VideoScorePredictor(nn.Module):
    def __init__(self, model_type='timesformer', variant='base', freeze_backbone=True, dropout_rate=0.5):
        super(VideoScorePredictor, self).__init__()
        self.model_type = model_type
        self.variant = variant
        self.freeze_backbone = freeze_backbone
        
        # Initialize the backbone model
        if model_type == 'timesformer':
            self._initialize_timesformer()
        elif model_type == 'video_swin':
            self._initialize_video_swin()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'timesformer' or 'video_swin'")
        
        # Create regression head
        self._create_regression_head(dropout_rate)
        
        # Apply backbone freezing if specified
        if freeze_backbone:
            self._freeze_backbone_layers()
    
    def _initialize_timesformer(self):
        """Initialize TimeSformer backbone with pre-trained weights."""
        print(f"Loading TimeSformer-{self.variant} backbone...")
        
        # Choose model variant
        if self.variant == 'base':
            model_name = "facebook/timesformer-base-finetuned-k400"
            self.hidden_size = 768
        elif self.variant == 'large':
            # Note: Using base model for now, can be updated when large is available
            model_name = "facebook/timesformer-base-finetuned-k400"
            self.hidden_size = 768
        else:
            raise ValueError(f"Unsupported TimeSformer variant: {self.variant}")
        
        # Load the pre-trained model
        self.backbone = TimesformerModel.from_pretrained(model_name)
        
        print(f"✓ TimeSformer backbone loaded successfully")
    
    def _initialize_video_swin(self):
        """Initialize Video Swin Transformer backbone with pre-trained weights."""
        print(f"Loading Video Swin Transformer-{self.variant} backbone...")
        
        # For now, use VideoMAE as a placeholder for Video Swin
        # This can be updated when Video Swin is available in transformers
        if self.variant == 'base':
            model_name = "MCG-NJU/videomae-base"
            self.hidden_size = 768
        else:
            raise ValueError(f"Unsupported Video Swin variant: {self.variant}")
        
        # Load the pre-trained model
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        
        print(f"✓ Video Swin backbone loaded successfully")
    
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
        
        print("✓ Regression head created successfully")
    
    def _freeze_backbone_layers(self):
        """Freeze backbone layers for transfer learning."""
        print("Freezing backbone layers...")
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        print("✓ Backbone layers frozen successfully")
    
    def forward(self, pixel_values, attention_mask=None):
        """
        Forward pass for score prediction.
        
        Args:
            pixel_values: Input tensor of shape [batch_size, num_frames, channels, height, width]
            attention_mask: Boolean tensor of shape [batch_size, num_frames] indicating real vs padded frames
            
        Returns:
            scores: Predicted scores tensor of shape [batch_size, 1] with values in range [0.0, 10.0]
        """
        
        print(f"Input tensor shape: {pixel_values.shape}")
        
        # Forward pass through backbone
        if self.model_type == 'timesformer':
            outputs = self.backbone(pixel_values=pixel_values)  # TimeSformer doesn't use attention masks
            # Get the pooled output (CLS token representation)
            features = outputs.last_hidden_state[:, 0]  # Shape: [batch_size, hidden_size]
            
        elif self.model_type == 'video_swin':
            outputs = self.backbone(pixel_values=pixel_values, attention_mask=attention_mask)
            # Get the pooled output
            features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        
        # Forward pass through regression head
        scores = self.regression_head(features)
        
        # Scale scores from [0, 1] to [0, 10]
        scores = scores * 10.0
        
        return scores.squeeze(-1)  # Remove last dimension to get [batch_size] 