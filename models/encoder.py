import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from configuration.config import settings

class MothEncoder(nn.Module):
    def __init__(self, message_length=settings.MESSAGE_LENGTH, alpha=0.7, max_adaptive_scale=0.001):
        """
        Moth Encoder for audio steganography
        Uses settings.INPUT_SIZE and settings.HIDDEN_SIZE for dimensions.
        Accepts alpha and max_adaptive_scale hyperparameters.
        Predicts an adaptive scale for the perturbation.
        """
        super(MothEncoder, self).__init__()
        self.alpha = alpha
        self.max_adaptive_scale = max_adaptive_scale
        self.message_length = message_length
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
        
        # Message embedding layer
        self.message_embedding = nn.Linear(message_length, hidden_size)
        
        # Feature processing
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        
        # Combined layers for perturbation
        self.fc_out = nn.Linear(hidden_size * 2, input_size)
        
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
    
    def forward(self, x, message=None, adaptive_scale=0.001, max_adaptive_scale=None, perceptual_loss_type='mse'):
        """
        Forward pass through the encoder model
        Args:
            x: Input audio tensor of shape (batch_size, 1, audio_length)
            message: Binary message tensor of shape (batch_size, message_length)
                    If None, a random message is generated
            adaptive_scale: Base strength of the watermark (will be adjusted adaptively)
            max_adaptive_scale: Maximum allowed adaptive scale value, defaults to self.max_adaptive_scale
            perceptual_loss_type: Type of perceptual loss to use ('mse', 'spec', or 'mel')
        Returns:
            watermarked: Watermarked audio tensor
            message: Binary message that was encoded
            perturbation: The perturbation applied to the original audio
            scale: The actual scale factor used for this batch
            perceptual_loss: The perceptual loss between original and watermarked audio
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Add debugging info
        input_size = x.shape[-1]
        print(f"DEBUG: Input audio shape: {x.shape}, Input size: {input_size}")
        
        # Use instance max_adaptive_scale if not provided
        if max_adaptive_scale is None:
            max_adaptive_scale = self.max_adaptive_scale
            
        # Ensure max_adaptive_scale is a tensor
        if not isinstance(max_adaptive_scale, torch.Tensor):
            max_adaptive_scale = torch.tensor(max_adaptive_scale, device=device)
        
        # Normalize input to improve stability
        x = torch.clamp(x, min=-0.99, max=0.99)
        
        # Generate random message if none provided
        if message is None:
            message = torch.randint(0, 2, (batch_size, self.message_length), dtype=torch.float32, device=device)
        
        # Extract features from the cover audio
        x_features = x  # Store original for residual connection
        print(f"DEBUG: Before conv1, x shape: {x.shape}")
        features = F.relu(self.conv1(x))
        print(f"DEBUG: After conv1, features shape: {features.shape}")
        features = F.relu(self.conv2(features))
        print(f"DEBUG: After conv2, features shape: {features.shape}")
        features = F.relu(self.conv3(features))
        print(f"DEBUG: After conv3, features shape: {features.shape}")
        
        # Flatten features
        features_flat = features.reshape(batch_size, -1)  # Using reshape instead of view for safety
        print(f"DEBUG: Flattened features shape: {features_flat.shape}")
        
        # Map features to intermediate representation
        features = F.relu(self.fc1(features_flat))
        print(f"DEBUG: After fc1, features shape: {features.shape}")
        
        # Generate perturbation from message and features
        message_features = F.relu(self.message_embedding(message.reshape(batch_size, -1)))
        print(f"DEBUG: Message features shape: {message_features.shape}")
        combined = torch.cat([features, message_features], dim=1)
        print(f"DEBUG: Combined features shape: {combined.shape}")
        
        # Generate perturbation pattern - prevent extreme values
        perturbation = torch.tanh(self.fc_out(combined))
        print(f"DEBUG: Generated perturbation shape: {perturbation.shape}")
        
        # Reshape to match audio
        perturbation = perturbation.reshape(batch_size, 1, -1)
        print(f"DEBUG: Reshaped perturbation shape: {perturbation.shape}")
        
        # Ensure perturbation is the right size (match audio length)
        if perturbation.shape[2] < x.shape[2]:
            # Pad if needed
            padding_size = x.shape[2] - perturbation.shape[2]
            perturbation = F.pad(perturbation, (0, padding_size))
            print(f"DEBUG: After padding, perturbation shape: {perturbation.shape}")
        elif perturbation.shape[2] > x.shape[2]:
            # Truncate if needed
            perturbation = perturbation[:, :, :x.shape[2]]
            print(f"DEBUG: After truncating, perturbation shape: {perturbation.shape}")
        
        # For now, use MSE for all perceptual loss types to avoid shape issues
        # This is temporary for debugging
        print(f"DEBUG: Using MSE loss instead of {perceptual_loss_type} to avoid shape issues")
        perceptual_loss = F.mse_loss(x, x + 0.001 * perturbation, reduction='none')
        perceptual_loss = perceptual_loss.mean(dim=(1, 2))  # Average across channels and time
        
        # Apply gradient stopping to prevent backprop through the adaptive scaling computation
        perceptual_loss_detached = perceptual_loss.detach()
        
        # Calculate adaptive scale factor
        # Start with a small scale and adjust based on perceptual properties
        with torch.no_grad():
            # Calculate scale factors that decrease as perceptual loss increases
            # Normalize perceptual loss to range [0, 1]
            normalized_loss = torch.clamp(perceptual_loss_detached / 0.1, min=0.0, max=1.0)
            # Inverse relationship: higher loss → lower scale
            loss_factor = 1.0 - normalized_loss
            
            # Apply a smooth scale adjustment
            scale = adaptive_scale * loss_factor
            
            # Ensure scale stays within bounds and has no NaN values
            scale = torch.clamp(scale, min=0.0001, max=max_adaptive_scale)
            scale = torch.nan_to_num(scale, nan=0.0001)
            
            # Reshape for broadcasting
            scale = scale.view(batch_size, 1, 1)
        
        # Apply scaled perturbation to create watermarked audio
        watermarked = x + scale * perturbation
        
        # Final clipping to ensure values stay in valid range
        watermarked = torch.clamp(watermarked, min=-1.0, max=1.0)
        
        # Final check for NaN values - replace with original audio if NaN is detected
        if torch.isnan(watermarked).any():
            watermarked = torch.where(torch.isnan(watermarked), x, watermarked)
            
        return watermarked, message, perturbation, scale.squeeze().detach(), perceptual_loss

    def _compute_perceptual_loss(self, original, perturbation, loss_type='mse'):
        """
        Compute perceptual loss between original and perturbed audio
        Args:
            original: Original audio tensor
            perturbation: Perturbation tensor
            loss_type: Type of perceptual loss to use ('mse', 'spec', 'mel')
        Returns:
            perceptual_loss: The perceptual loss between original and perturbed audio
        """
        # Small temporary scale to calculate perceptual impact
        temp_scale = 0.001
        perturbed = original + temp_scale * perturbation
        perturbed = torch.clamp(perturbed, min=-1.0, max=1.0)
        
        batch_size = original.shape[0]
        
        # Calculate different types of perceptual loss
        if loss_type == 'mse':
            # Mean squared error in time domain
            loss = F.mse_loss(original, perturbed, reduction='none')
            loss = loss.mean(dim=(1, 2))  # Average across channels and time
            
        elif loss_type == 'spec':
            # Spectral loss
            try:
                # Use spectrogram for perceptual loss
                spec_transform = torchaudio.transforms.Spectrogram().to(original.device)
                orig_spec = spec_transform(original.squeeze(1))
                pert_spec = spec_transform(perturbed.squeeze(1))
                
                # Compute spectral loss
                loss = F.mse_loss(orig_spec, pert_spec, reduction='none')
                loss = loss.mean(dim=(1, 2))  # Average across frequency and time
            except Exception as e:
                # Log the error for debugging
                print(f"Spectrogram loss failed: {e}. Falling back to MSE.")
                # Fallback to MSE if spectral loss fails
                loss = F.mse_loss(original, perturbed, reduction='none')
                loss = loss.mean(dim=(1, 2))
                
        elif loss_type == 'mel':
            # Mel-spectrogram loss
            try:
                # Create a custom MelSpectrogram transform with appropriate parameters
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=settings.SAMPLE_RATE,
                    n_fft=settings.FFT_SIZE,
                    hop_length=settings.HOP_LEN,
                    n_mels=settings.N_MELS
                ).to(original.device)
                
                # Generate mel spectrograms
                orig_mel = mel_transform(original.squeeze(1))
                pert_mel = mel_transform(perturbed.squeeze(1))
                
                # Use log mel spectrogram for better perceptual correlation
                orig_mel = torch.log(orig_mel + 1e-5)
                pert_mel = torch.log(pert_mel + 1e-5)
                
                # MSE in mel-spectrogram domain - don't rely on feeding this to neural network layers
                mel_mse = F.mse_loss(orig_mel, pert_mel, reduction='none')
                loss = mel_mse.mean(dim=(1, 2))  # Average across mel bands and time

            except Exception as e:
                # Log the error for debugging
                print(f"Mel spectrogram loss failed: {e}. Falling back to MSE.")
                # Fallback to MSE if mel loss fails
                loss = F.mse_loss(original, perturbed, reduction='none')
                loss = loss.mean(dim=(1, 2))
        else:
            # Default to MSE
            loss = F.mse_loss(original, perturbed, reduction='none')
            loss = loss.mean(dim=(1, 2))
        
        # Ensure no NaN values
        loss = torch.nan_to_num(loss, nan=0.1)
        
        return loss

    def compute_loss(self, original, watermarked, message, decoder_output, perceptual_loss=None, alpha=0.5):
        """
        Compute the combined loss for the watermarking model
        
        Args:
            original: Original audio tensor (batch_size, 1, audio_length)
            watermarked: Watermarked audio tensor (batch_size, 1, audio_length)
            message: Binary message tensor (batch_size, message_length)
            decoder_output: Output from decoder (batch_size, message_length)
            perceptual_loss: Pre-computed perceptual loss (optional)
            alpha: Weight for detection loss (0-1), where higher means more emphasis on detection
                  
        Returns:
            Dictionary containing:
                - total_loss: Combined loss for backpropagation
                - perceptual_loss: Loss measuring audio quality degradation
                - detection_loss: Loss measuring message detection accuracy
                - alpha: The weight used for balancing the losses
        """
        # Compute perceptual loss if not provided
        if perceptual_loss is None:
            # Use simple MSE as perceptual loss to avoid shape/NaN issues
            perceptual_loss = F.mse_loss(original, watermarked, reduction='none').mean(dim=(1, 2))
        
        # Compute detection loss (binary cross entropy)
        detection_loss = F.binary_cross_entropy_with_logits(
            decoder_output, message, reduction='none'
        ).mean(dim=1)  # Mean across message bits
        
        # Check for NaN values and replace with reasonable defaults
        if torch.isnan(perceptual_loss).any():
            print("WARNING: NaN values detected in perceptual loss, using default value")
            perceptual_loss = torch.where(
                torch.isnan(perceptual_loss), 
                torch.tensor(0.1, device=perceptual_loss.device), 
                perceptual_loss
            )
            
        if torch.isnan(detection_loss).any():
            print("WARNING: NaN values detected in detection loss, using default value")
            detection_loss = torch.where(
                torch.isnan(detection_loss), 
                torch.tensor(0.5, device=detection_loss.device), 
                detection_loss
            )
        
        # Get mean losses across batch
        batch_perceptual_loss = perceptual_loss.mean()
        batch_detection_loss = detection_loss.mean()
        
        # Combine losses: perceptual loss (audio quality) and detection loss (message accuracy)
        # alpha controls the balance between the two losses
        total_loss = (1 - alpha) * batch_perceptual_loss + alpha * batch_detection_loss
        
        # Return all losses and metrics for monitoring
        return {
            'total_loss': total_loss,
            'perceptual_loss': batch_perceptual_loss.detach(),
            'detection_loss': batch_detection_loss.detach(),
            'alpha': alpha
        }