import torch
import torch.nn as nn
import torchvision.models as models

class SurfManeuverModel(nn.Module):
    def __init__(self, hidden_size=128, num_classes=7, mode='dev'):  # Set num_classes to the total number of maneuver types
        super(SurfManeuverModel, self).__init__()
        self.mode = mode
        
        # Pretrained CNN for feature extraction
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if self.mode == 'dev':
            cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Change input channels to 1 for grayscale
        self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])  # Remove the final classification layer
        cnn_out_dim = cnn.fc.in_features

        # LSTM layer
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=hidden_size, batch_first=True)
        
        # Fully connected layer for final classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        
        # Extract features for each frame in the sequence
        cnn_features = []
        for i in range(seq_len):
            features = self.feature_extractor(x[:, i, :, :, :])  # Pass each frame through CNN
            features = features.view(batch_size, -1)  # Flatten
            cnn_features.append(features)

        # Stack CNN features and pass through LSTM
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, cnn_out_dim)
        lstm_out, _ = self.lstm(cnn_features)
        
        # Take the final LSTM output for classification
        out = self.fc(lstm_out[:, -1, :])
        return out
