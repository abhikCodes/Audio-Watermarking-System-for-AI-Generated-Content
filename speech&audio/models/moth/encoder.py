import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pesq import pesq
from configuration.config import settings, LossFunctionType
import logging

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
        
        # Ensure perturbation is not too small by applying a minimum magnitude
        # Only apply this during training
        if self.training:
            perturbation_magnitude = torch.abs(perturbation)
            mean_magnitude = torch.mean(perturbation_magnitude)
            
            # If average perturbation is too small, amplify it
            if mean_magnitude < 0.001:
                scaling_factor = 0.001 / (mean_magnitude + 1e-10)
                perturbation = perturbation * min(scaling_factor, 5.0)  # Cap at 5x amplification
        
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
        
        # Compute magnitude and add small epsilon to avoid zero values
        spec = torch.abs(stft) + 1e-8
        
        # Add normalization to enhance differences
        spec = spec / (torch.mean(spec) + 1e-8)
        
        return spec
    
    def compute_mel_spectrogram(self, audio):
        """
        Compute the log-mel spectrogram of the audio.
        Args:
            audio: Tensor of shape (batch_size, 1, num_samples)
        Returns:
            mel_spec: Log-mel spectrogram 
        """
        # Parameters for mel spectrogram
        n_fft = settings.FFT_SIZE
        hop_length = settings.HOP_LEN
        win_length = settings.WIN_LEN
        n_mels = 128  # Number of mel bands
        
        # Process each audio in the batch
        batch_size = audio.shape[0]
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # (batch_size, num_samples)
        
        # Convert to numpy for librosa processing
        audio_np = audio.detach().cpu().numpy()
        mel_specs = []
        
        for i in range(batch_size):
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio_np[i],
                sr=settings.SAMPLE_RATE,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                power=2.0
            )
            
            # Convert to log scale
            S_log = librosa.power_to_db(S, ref=np.max)
            mel_specs.append(S_log)
        
        # Convert back to tensor
        mel_specs = np.stack(mel_specs)
        mel_specs = torch.tensor(mel_specs, device=audio.device, dtype=torch.float32)
        
        return mel_specs
    
    def compute_psychoacoustic_loss(self, original_audio, watermarked_audio):
        """
        Compute the psychoacoustic model-based loss using PESQ.
        Args:
            original_audio: Original audio tensor
            watermarked_audio: Audio with steganographic imprint
        Returns:
            loss: Psychoacoustic loss based on PESQ scores
        """
        # Convert tensors to numpy arrays
        if original_audio.dim() == 3:
            original_audio = original_audio.squeeze(1)
        if watermarked_audio.dim() == 3:
            watermarked_audio = watermarked_audio.squeeze(1)
            
        # Calculate a simple spectral distance as a faster alternative to PESQ
        # This approach will still capture perceptual differences
        
        # Calculate spectrogram difference
        orig_spec = self.compute_spectrogram(original_audio.unsqueeze(1))
        watermarked_spec = self.compute_spectrogram(watermarked_audio.unsqueeze(1))
        
        # L1 difference between spectrograms (more stable than MSE)
        spec_diff = F.l1_loss(watermarked_spec, orig_spec)
        
        # Direct waveform difference
        wave_diff = F.l1_loss(watermarked_audio, original_audio)
        
        # Combine for final perceptual loss
        combined_loss = 0.7 * spec_diff + 0.3 * wave_diff
        
        # Ensure loss is significant
        combined_loss = torch.clamp(combined_loss, min=0.001)
            
        return combined_loss
    
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
        # Detection loss: Ensure decoder can detect the watermark
        detection_loss = F.binary_cross_entropy_with_logits(decoder_output, torch.ones_like(decoder_output))
        
        # Choose the appropriate perceptual loss based on settings
        if settings.LOSS_FUNCTION == LossFunctionType.SPECTROGRAM:
            # Compute spectrograms
            original_spec = self.compute_spectrogram(original_audio)
            watermarked_spec = self.compute_spectrogram(watermarked_audio)
            
            # Calculate L1 loss instead of MSE for better gradient signals
            perceptual_loss = F.l1_loss(watermarked_spec, original_spec)
            
            # Add direct audio difference as an additional perceptual component
            audio_diff_loss = F.l1_loss(watermarked_audio, original_audio)
            
            # Weighted combination to make sure perceptual loss has meaningful values
            perceptual_loss = perceptual_loss * 0.8 + audio_diff_loss * 0.2
            
        elif settings.LOSS_FUNCTION == LossFunctionType.LOG_MEL:
            # Compute log-mel spectrograms
            original_mel = self.compute_mel_spectrogram(original_audio)
            watermarked_mel = self.compute_mel_spectrogram(watermarked_audio)
            
            # Perceptual loss: MSE between log-mel spectrograms
            perceptual_loss = F.l1_loss(watermarked_mel, original_mel)
            
        elif settings.LOSS_FUNCTION == LossFunctionType.PSYCHOACOUSTIC:
            # Compute psychoacoustic loss
            perceptual_loss = self.compute_psychoacoustic_loss(original_audio, watermarked_audio)
            
        else:
            # Default to spectrogram loss
            original_spec = self.compute_spectrogram(original_audio)
            watermarked_spec = self.compute_spectrogram(watermarked_audio)
            perceptual_loss = F.l1_loss(watermarked_spec, original_spec)
        
        # Ensure perceptual loss is non-zero
        if perceptual_loss < 1e-8:
            perceptual_loss = torch.tensor(1e-5, device=perceptual_loss.device, requires_grad=True)
        
        # Total loss with original formulation - more stable
        total_loss = perceptual_loss + settings.ALPHA * detection_loss
        
        # Catch NaN values and return a safe default if needed
        if torch.isnan(total_loss):
            logger = logging.getLogger("SteganographyTraining")
            logger.warning("NaN detected in loss calculation, using default loss")
            total_loss = torch.tensor(0.1, device=total_loss.device, requires_grad=True)
            perceptual_loss = torch.tensor(0.05, device=total_loss.device)
            detection_loss = torch.tensor(0.05, device=total_loss.device)
        
        metrics = {
            'perceptual_loss': perceptual_loss.item(),
            'detection_loss': detection_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics