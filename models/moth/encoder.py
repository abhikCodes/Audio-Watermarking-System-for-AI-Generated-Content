import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pesq import pesq
from configuration.config import settings

class MothEncoder(nn.Module):
    def __init__(self):
        """
        Moth Encoder for audio steganography
        Uses settings.INPUT_SIZE and settings.HIDDEN_SIZE for dimensions.
        """
        super(MothEncoder, self).__init__()
        input_size = settings.INPUT_SIZE
        hidden_size = settings.HIDDEN_SIZE

        # Convolutional feature extractor
        k = settings.CONV_KERNEL_SIZE
        s = settings.CONV_STRIDE
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=k, stride=s)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=k, stride=s)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=k, stride=s)
        
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
        perturbation = torch.tanh(self.fc2(x)) * settings.PERTURB_SCALE
        
        return perturbation.view(-1, 1, perturbation.size(1))
    
    def compute_spectrogram(self, audio):
        """
        Compute the magnitude spectrogram of the audio.
        Args:
            audio: Tensor of shape (batch_size, 1, num_samples)
        Returns:
            spec: Magnitude spectrogram of shape (batch_size, freq_bins, time_steps)
        """
        n_fft = settings.FFT_SIZE
        hop_length = settings.HOP_LEN
        win_length = settings.WIN_LEN
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # (batch_size, num_samples)
        
        stft = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(audio.device),
            return_complex=True
        )
        
        spec = torch.abs(stft)
        return spec
    
    def compute_loss(self, original_audio, watermarked_audio, decoder_output):
        """
        Compute the loss for training using spectrogram-based perceptual loss.
        Args:
            original_audio: Original audio tensor
            watermarked_audio: Audio with steganographic imprint
            decoder_output: Output from the decoder
        Returns:
            total_loss: Combined loss value
            metrics: Dictionary of individual loss components
        """
        # Compute spectrograms
        original_spec = self.compute_spectrogram(original_audio)
        watermarked_spec = self.compute_spectrogram(watermarked_audio)
        
        # Perceptual loss: MSE between spectrograms
        perceptual_loss = F.mse_loss(watermarked_spec, original_spec)
        
        # Detection loss: Ensure decoder can detect the watermark
        detection_loss = F.binary_cross_entropy_with_logits(decoder_output, torch.ones_like(decoder_output))
        
        # Total loss with balancing factor
        total_loss = perceptual_loss + settings.ALPHA * detection_loss
        
        metrics = {
            'perceptual_loss': perceptual_loss.item(),
            'detection_loss': detection_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics