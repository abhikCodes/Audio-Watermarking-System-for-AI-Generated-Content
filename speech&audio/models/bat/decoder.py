import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configuration.config import settings

class BatDecoder(nn.Module):
    def __init__(self):
        """
        Bat Decoder for audio steganography detection
        Args:
            input_size: Size of input audio segment (default: 96000 for 6 seconds at 16kHz)
            hidden_size: Size of hidden layers
        """
        super(BatDecoder, self).__init__()
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
        x = torch.randn(1, 1, input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
        Forward pass to detect steganographic imprint
        Args:
            x: Input audio tensor of shape (batch_size, 1, input_size)
        Returns:
            detection: Probability of steganographic imprint presence
        """
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Detection
        x = F.relu(self.fc1(x))
        detection = torch.sigmoid(self.fc2(x))
        
        return detection
    
    def compute_loss(self, predictions, targets):
        """
        Compute the loss for training
        Args:
            predictions: Model predictions (0 for clean audio, 1 for watermarked)
            targets: Ground truth labels
        Returns:
            loss: Binary cross entropy loss
            accuracy: Detection accuracy
        """
        loss = F.binary_cross_entropy(predictions, targets)
        accuracy = ((predictions > 0.5) == targets).float().mean()
        
        return loss, accuracy.item() 