import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pesq import pesq

class MothEncoder(nn.Module):
    def __init__(self, input_size=96000, hidden_size=512):
        """
        Moth Encoder for audio steganography
        Args:
            input_size: Size of input audio segment (default: 96000 for 6 seconds at 16kHz)
            hidden_size: Size of hidden layers
        """
        super(MothEncoder, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, stride=4)
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size(input_size)
        
        # Perturbation generation layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        
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
        Forward pass to generate steganographic perturbation
        Args:
            x: Input audio tensor of shape (batch_size, 1, input_size)
        Returns:
            perturbation: Generated perturbation to be added to original audio
        """
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Generate perturbation
        x = F.relu(self.fc1(x))
        perturbation = torch.tanh(self.fc2(x)) * 0.003
        
        return perturbation.view(-1, 1, perturbation.size(1))
    
    def compute_loss(self, original_audio, watermarked_audio, decoder_output):
        """
        Compute the loss for training
        Args:
            original_audio: Original audio tensor
            watermarked_audio: Audio with steganographic imprint
            decoder_output: Output from the Bat decoder
        Returns:
            total_loss: Combined loss value
            metrics: Dictionary of individual loss components
        """
        # Perceptual loss (MSE proxy for PESQ)
        perceptual_loss = F.mse_loss(watermarked_audio, original_audio)
        
        # Detection loss (ensure decoder can detect the watermark)
        detection_loss = F.binary_cross_entropy_with_logits(decoder_output, torch.ones_like(decoder_output))
        
        # Total loss with balancing factor
        alpha = 0.1  # Balance between perceptual and detection loss
        total_loss = perceptual_loss + alpha * detection_loss
        
        metrics = {
            'perceptual_loss': perceptual_loss.item(),
            'detection_loss': detection_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics 