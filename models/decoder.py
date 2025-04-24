import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configuration.config import settings

class BatDecoder(nn.Module):
    def __init__(self, detection_threshold=0.5):
        """
        Bat Decoder for audio steganography detection
        Args:
            input_size: Size of input audio segment (default: 96000 for 6 seconds at 16kHz)
            hidden_size: Size of hidden layers
            detection_threshold: Threshold for classifying as watermarked.
        """
        super(BatDecoder, self).__init__()
        self.detection_threshold = detection_threshold # Store threshold
        input_size = settings.DECODER_INPUT_SIZE
        hidden_size = settings.DECODER_HIDDEN_SIZE

        # Shared conv feature extractor
        k = settings.CONV_KERNEL_SIZE
        s = settings.CONV_STRIDE
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=k, stride=s)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=k, stride=s)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=k, stride=s)
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size(input_size)
        
        # Detection layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_conv_output_size(self, input_size):
        """Calculate the size of the output after convolutions"""
        # Temporarily create layers to calculate size
        conv1 = nn.Conv1d(1, 32, kernel_size=settings.CONV_KERNEL_SIZE, stride=settings.CONV_STRIDE)
        conv2 = nn.Conv1d(32, 64, kernel_size=settings.CONV_KERNEL_SIZE, stride=settings.CONV_STRIDE)
        conv3 = nn.Conv1d(64, 128, kernel_size=settings.CONV_KERNEL_SIZE, stride=settings.CONV_STRIDE)
        with torch.no_grad():
            x = torch.randn(1, 1, input_size)
            x = F.relu(conv1(x))
            x = F.relu(conv2(x))
            x = F.relu(conv3(x))
        return int(np.prod(x.size()))
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the decoder model
        Args:
            x: Input audio tensor of shape (batch_size, 1, audio_length)
        Returns:
            Binary prediction tensor of shape (batch_size, 1)
        """
        # Normalize input to prevent extreme values
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Apply convolutional layers
        features = F.relu(self.conv1(x))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))
        
        # Global average pooling
        features = torch.mean(features, dim=2)
        
        # Fully connected layers
        features = F.relu(self.fc1(features))
        features = self.fc2(features)
        
        # Sanity check - replace any NaN values that might occur
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
            
        return features
        
    def compute_loss(self, pred_logits, targets):
        """
        Compute binary cross entropy loss and accuracy for the detector
        Args:
            pred_logits: Prediction logits from the detector of shape (batch_size, 1)
            targets: Target labels (0 for clean, 1 for watermarked) of shape (batch_size, 1)
        Returns:
            loss: Binary cross entropy loss
            accuracy: Detector accuracy
        """
        # Use BCE with logits for numerical stability
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Make sure pred_logits and targets have the same shape
        if pred_logits.shape != targets.shape:
            pred_logits = pred_logits.view(-1, 1)
            targets = targets.view(-1, 1)
        
        # Calculate the loss
        loss = loss_fn(pred_logits, targets)
        
        # Get predicted binary labels (0 or 1)
        # Apply sigmoid to convert logits to probabilities
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs >= self.detection_threshold).float()
        
        # Calculate accuracy
        accuracy = (pred_binary == targets).float().mean()
        
        return loss, accuracy