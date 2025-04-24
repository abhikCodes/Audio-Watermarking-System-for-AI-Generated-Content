import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import numpy as np
import librosa
import os
import time
import datetime
import csv
from tqdm import tqdm
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.encoder import MothEncoder
from models.decoder import BatDecoder
from configuration.config import settings
from training.data_utils import get_train_test_filepaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.TRAIN_LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SteganographyTraining")

# --- Auto-detect device ---
def get_default_device():
    if settings.DEVICE:
        logger.info(f"Using configured device from settings: {settings.DEVICE}")
        return settings.DEVICE
    elif torch.cuda.is_available():
        logger.info("CUDA detected. Using GPU.")
        return 'cuda'
    elif torch.backends.mps.is_available():
        logger.info("MPS detected. Using Apple Silicon GPU.")
        return 'mps'
    else:
        logger.info("No GPU detected. Using CPU.")
        return 'mps'
# --------------------------

class AudioDataset(Dataset):
    def __init__(self, audio_paths, base_dir=None, max_samples=None):
        """
        Dataset for audio steganography
        Args:
            audio_paths: List of audio file paths (absolute or relative to base_dir)
            base_dir: Base directory for relative paths (if not absolute)
            max_samples: Max number of samples to load (for testing)
        """
        self.base_dir = base_dir if base_dir else ""
        self.audio_paths = []
        self.max_length = settings.MAX_LENGTH
        self.sample_rate = settings.SAMPLE_RATE
        self.failed_loads = 0 # Keep track of failed loads

        # Handle input paths - either list or directory
        if isinstance(audio_paths, list):
            logger.info(f"Using provided list of {len(audio_paths)} audio files")
            potential_paths = audio_paths
        else:
            # For backward compatibility - if a directory is passed instead of paths
            audio_dir = audio_paths
            logger.info(f"Scanning directory {audio_dir} for WAV files...")
            potential_paths = []
            for root, _, files in os.walk(audio_dir):
                for file in files:
                    if file.endswith('.wav'):
                        potential_paths.append(os.path.join(root, file))

        # Limit samples if max_samples is set
        if max_samples is not None and len(potential_paths) > max_samples:
            logger.info(f"Limiting dataset to {max_samples} samples for testing.")
            # Optional: shuffle before selecting subset? For now, just take the first N
            self.audio_paths = potential_paths[:max_samples]
        else:
            self.audio_paths = potential_paths

        logger.info(f"Dataset created with {len(self.audio_paths)} WAV files")

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # If base_dir is set and path is not absolute, join with base_dir
        if self.base_dir and not os.path.isabs(audio_path):
            full_path = os.path.join(self.base_dir, audio_path)
        else:
            full_path = audio_path
            
        # Add logging before attempting to load
        logger.debug(f"Attempting to load audio file: {full_path}") 
        
        try:
            audio, _ = librosa.load(full_path, sr=self.sample_rate, mono=True)
            audio = self.normalize_audio(audio)
            audio = torch.from_numpy(audio).float()
            if len(audio) < self.max_length:
                audio = torch.nn.functional.pad(audio, (0, self.max_length - len(audio)))
            else:
                audio = audio[:self.max_length]
            return audio.unsqueeze(0)
        except Exception as e:
            logger.warning(f"Could not load audio file: {full_path}. Error: {e}")
            self.failed_loads += 1
            return None
    
    def normalize_audio(self, audio):
        """Normalize audio to range [-1, 1]"""
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio
    
    def reduce_noise(self, audio, threshold=0.01):
        """Simple noise gate to reduce background noise"""
        audio_abs = np.abs(audio)
        mask = audio_abs > threshold
        return audio * mask

# --- Custom Collate Function ---
def collate_fn_skip_none(batch):
    """Collate function that filters out None values."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)
# -----------------------------

def train_epoch(train_loader, encoder, decoder, optimizer, device, config):
    """Train the model for one epoch"""
    encoder.train()
    decoder.train()
    
    # Statistics tracking
    epoch_encoder_loss = 0
    epoch_decoder_loss = 0
    epoch_decoder_accuracy = 0
    epoch_perceptual_loss = 0
    epoch_detection_loss = 0
    epoch_scale_factor = 0
    
    # Track batches with NaN values
    nan_batches = 0
    total_batches = len(train_loader)
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, audio_batch in enumerate(progress_bar):
        try:
            # Move data to the appropriate device
            audio = audio_batch.to(device)
            batch_size = audio.shape[0]
            
            # Zero the gradients
            optimizer['encoder'].zero_grad()
            optimizer['decoder'].zero_grad()
            
            # Step 1: Generate watermarked audio
            watermarked, message, perturbation, scale, perceptual_loss = encoder(
                audio, 
                adaptive_scale=config['adaptive_scale'],
                max_adaptive_scale=config['max_adaptive_scale'],
                perceptual_loss_type=config['perceptual_loss_type']
            )
            
            # Detect NaN values early
            if torch.isnan(watermarked).any():
                logging.warning(f"NaN values detected in watermarked audio. Scale: {scale.mean().item()}")
                watermarked = torch.nan_to_num(watermarked)
                
            # Scale is a per-batch value, calculate mean for reporting
            avg_scale = scale.mean().item()
            
            # Create detector targets (0 for original, 1 for watermarked)
            detector_targets_orig = torch.zeros((batch_size, 1), device=device)
            detector_targets_wm = torch.ones((batch_size, 1), device=device)
            
            # Step 2: Pass original and watermarked through detector
            detector_pred_orig = decoder(audio)
            detector_pred_wm = decoder(watermarked)
            
            # Check for NaN in decoder output
            if torch.isnan(detector_pred_orig).any() or torch.isnan(detector_pred_wm).any():
                logging.warning("NaN values detected in decoder output. Skipping batch.")
                nan_batches += 1
                continue
                
            # Step 3: Calculate detection loss and accuracy for both original and watermarked
            detection_loss_orig, acc_orig = decoder.compute_loss(detector_pred_orig, detector_targets_orig)
            detection_loss_wm, acc_wm = decoder.compute_loss(detector_pred_wm, detector_targets_wm)
            
            # Combined detection loss (average of original and watermarked)
            detection_loss = (detection_loss_orig + detection_loss_wm) / 2
            
            # Combined detection accuracy (average of original and watermarked)
            detection_accuracy = (acc_orig + acc_wm) / 2
            
            # Scale perceptual_loss for better balance with detection_loss
            perceptual_loss_scaled = perceptual_loss * 100.0
            
            # Alpha balances perceptual loss and detection loss
            # Higher alpha → more focus on detection, lower alpha → more focus on perceptual quality
            encoder_loss = (1 - config['alpha']) * perceptual_loss_scaled + config['alpha'] * detection_loss
            
            # Step 4: Backpropagate and update model weights
            # Check for NaN in loss
            if torch.isnan(encoder_loss).any() or torch.isnan(detection_loss).any():
                logging.warning(f"NaN detected in loss calculation. Encoder: {torch.isnan(encoder_loss).any()}, "
                              f"Detector: {torch.isnan(detection_loss).any()}")
                nan_batches += 1
                continue
                
            # Backpropagate
            encoder_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer['encoder'].step()
            optimizer['decoder'].step()
            
            # Update statistics (use .item() to get Python number from tensor)
            epoch_encoder_loss += encoder_loss.item()
            epoch_decoder_loss += detection_loss.item()
            epoch_decoder_accuracy += detection_accuracy.item()
            epoch_perceptual_loss += perceptual_loss.item()
            epoch_detection_loss += detection_loss.item()
            epoch_scale_factor += avg_scale
            
            # Update progress bar
            progress_bar.set_postfix({
                'enc_loss': f"{encoder_loss.item():.4f}",
                'dec_loss': f"{detection_loss.item():.4f}",
                'dec_acc': f"{detection_accuracy.item():.4f}",
                'scale': f"{avg_scale:.4f}"
            })
            
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            nan_batches += 1
            continue
    
    # Calculate averages (accounting for batches with NaN that were skipped)
    processed_batches = total_batches - nan_batches
    if processed_batches > 0:
        epoch_encoder_loss /= processed_batches
        epoch_decoder_loss /= processed_batches
        epoch_decoder_accuracy /= processed_batches
        epoch_perceptual_loss /= processed_batches
        epoch_detection_loss /= processed_batches
        epoch_scale_factor /= processed_batches
    else:
        logging.error("All batches contained NaN values. Training failed for this epoch.")
        return {
            'encoder_loss': float('nan'),
            'decoder_loss': float('nan'),
            'decoder_accuracy': float('nan'),
            'perceptual_loss': float('nan'),
            'detection_loss': float('nan'),
            'scale_factor': float('nan'),
            'nan_batches': nan_batches
        }
    
    # Return epoch statistics
    return {
        'encoder_loss': epoch_encoder_loss,
        'decoder_loss': epoch_decoder_loss,
        'decoder_accuracy': epoch_decoder_accuracy,
        'perceptual_loss': epoch_perceptual_loss,
        'detection_loss': epoch_detection_loss,
        'scale_factor': epoch_scale_factor,
        'nan_batches': nan_batches
    }

def test_model(
    encoder: MothEncoder,
    decoder: BatDecoder,
    test_dataloader: DataLoader,
    device: torch.device,
    perceptual_loss_type: str,
) -> Dict[str, float]:
    """
    Evaluate encoder and decoder on test data with improved NaN handling.
    
    Args:
        encoder: Trained encoder model
        decoder: Trained decoder model
        test_dataloader: DataLoader for test data
        device: Device to run evaluation on
        perceptual_loss_type: Type of perceptual loss used
        
    Returns:
        Dictionary with test metrics
    """
    encoder.eval()
    decoder.eval()
    
    test_metrics = {
        'test_encoder_loss': 0.0,
        'test_decoder_loss': 0.0,
        'test_decoder_accuracy': 0.0,
        'test_perceptual_loss': 0.0,
        'test_detection_loss': 0.0,
        'test_snr': 0.0,
        'avg_test_scale_factor': 0.0,
    }
    
    test_batches_processed = 0
    total_scale_factor = 0.0
    nan_batches = 0
    total_batches = len(test_dataloader)
    
    with torch.no_grad():
        for audio_batch in test_dataloader:
            if audio_batch is None:
                continue
                
            try:
                audio = audio_batch.to(device)
                batch_size = audio.size(0)
                
                # Ensure audio has no NaN values
                if torch.isnan(audio).any():
                    audio = torch.nan_to_num(audio, nan=0.0)
                
                # Test decoder on original audio
                pred_original_logits = decoder(audio)
                loss_original, acc_original = decoder.compute_loss(
                    pred_original_logits,
                    torch.zeros(batch_size, 1).to(device)
                )
                
                # Create watermarked audio using the encoder's forward method
                # The signature is now watermarked, message, perturbation, scale_factor, perceptual_loss
                watermarked, _, perturbation, scale_factor, _ = encoder(
                    audio, 
                    adaptive_scale=0.001,  # Smaller default value for stability
                    max_adaptive_scale=0.005,  # Use arg value from training
                    perceptual_loss_type=perceptual_loss_type
                )
                
                # Update total scale factor for averaging 
                total_scale_factor += scale_factor.sum().item()
                test_batches_processed += batch_size
                
                # Test decoder on watermarked audio
                pred_watermarked_logits = decoder(watermarked)
                loss_watermarked, acc_watermarked = decoder.compute_loss(
                    pred_watermarked_logits,
                    torch.ones(batch_size, 1).to(device)
                )
                
                # Calculate combined test metrics
                decoder_test_loss = (loss_original + loss_watermarked) / 2
                decoder_test_acc = (acc_original + acc_watermarked) / 2
                
                # Calculate SNR (Signal-to-Noise Ratio)
                signal_power = torch.mean(audio ** 2, dim=(1, 2))
                noise_power = torch.mean((watermarked - audio) ** 2, dim=(1, 2))
                # Add epsilon to avoid division by zero
                batch_snr = 10 * torch.log10((signal_power + 1e-10) / (noise_power + 1e-10))
                avg_batch_snr = torch.mean(batch_snr).item()
                
                # Compute encoder loss with the compute_loss method
                encoder_test_loss_val, loss_metrics = encoder.compute_loss(
                    audio, watermarked, pred_watermarked_logits, 
                    perceptual_loss_type=perceptual_loss_type
                )
                
                # Update test metrics with batch values
                test_metrics['test_encoder_loss'] += encoder_test_loss_val.item() * batch_size
                test_metrics['test_decoder_loss'] += decoder_test_loss.item() * batch_size
                test_metrics['test_decoder_accuracy'] += decoder_test_acc * batch_size
                test_metrics['test_perceptual_loss'] += loss_metrics['perceptual_loss'] * batch_size
                test_metrics['test_detection_loss'] += loss_metrics['detection_loss'] * batch_size
                test_metrics['test_snr'] += avg_batch_snr * batch_size
                
            except Exception as e:
                logger.warning(f"Error in test batch: {str(e)}. Skipping.")
                nan_batches += 1
                continue
            
    # Calculate final averages
    if test_batches_processed > 0:
        test_metrics['test_encoder_loss'] /= test_batches_processed
        test_metrics['test_decoder_loss'] /= test_batches_processed
        test_metrics['test_decoder_accuracy'] /= test_batches_processed
        test_metrics['test_perceptual_loss'] /= test_batches_processed
        test_metrics['test_detection_loss'] /= test_batches_processed
        test_metrics['test_snr'] /= test_batches_processed
        test_metrics['avg_test_scale_factor'] = total_scale_factor / test_batches_processed
    else:
        logger.warning("No valid batches processed during testing.")
        for key in test_metrics:
            test_metrics[key] = 0.0

    # Log NaN statistics
    if nan_batches > 0:
        logger.warning(f"{nan_batches}/{total_batches} test batches skipped due to errors or NaN values.")

    return test_metrics

def create_results_csv(results, csv_path):
    """Create a CSV file with test results from all models."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Determine if file exists to decide whether to write headers
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as csvfile:
        # Get fieldnames from the first result dictionary
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write all results
            for result in results:
                writer.writerow(result)
            
            logging.info(f"Test results saved to {csv_path}")

def train_models(
    train_dir: str,
    output_dir: Path,
    learning_rate: float,
    alpha: float,
    max_adaptive_scale: float,
    perceptual_loss_type: str,
    num_epochs: int,
    batch_size: int,
    device_str: Optional[str] = None,
    is_test_mode: bool = False,
    test_samples: Optional[int] = None,
    test_split_ratio: float = 0.1,
    random_seed: int = 42,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.001
):
    """
    Train both Moth and Bat models with specified hyperparameters.
    Saves models and a README to the output_dir.
    Implements early stopping based on test decoder loss.
    Uses adaptive perturbation scaling up to max_adaptive_scale.
    """
    # --- Create Output Directory ---
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # --- Set seeds for reproducibility ---
    logger.info(f"Setting random seed to {random_seed} for reproducibility")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available(): # Note: MPS doesn't have manual_seed equivalent, relies on torch.manual_seed
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # No specific MPS seed setting, but torch.manual_seed should cover it.

    # --- Determine Device ---
    if device_str is None:
        device_str = get_default_device()
    try:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    except RuntimeError as e:
        logger.error(f"Could not use device '{device_str}'. Error: {e}. Falling back to CPU.")
        device = torch.device('cpu')

    # --- Log Training Parameters ---
    logger.info(f"Starting training run with parameters:")
    logger.info(f"- Training directory: {train_dir}")
    logger.info(f"- Output directory: {output_dir}")
    logger.info(f"- Learning Rate: {learning_rate}")
    logger.info(f"- Alpha (Detection Loss Weight): {alpha}")
    logger.info(f"- Max Adaptive Scale: {max_adaptive_scale}")
    logger.info(f"- Perceptual Loss Type: {perceptual_loss_type}")
    logger.info(f"- Number of epochs: {num_epochs}")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Test split ratio: {test_split_ratio}")
    logger.info(f"- Random seed: {random_seed}")
    logger.info(f"- Early Stopping Patience: {early_stopping_patience}")
    logger.info(f"- Early Stopping Min Delta: {early_stopping_min_delta}")
    logger.info(f"- Device: {device}")
    logger.info(f"- Test Mode: {is_test_mode}")
    if is_test_mode:
        logger.info(f"- Test Samples: {test_samples}")

    # --- Generate README ---
    readme_content = f"""# Training Run Configuration

- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Training Directory:** `{train_dir}`
- **Output Directory:** `{output_dir}`
- **Learning Rate:** `{learning_rate}`
- **Alpha (Detection Loss Weight):** `{alpha}`
- **Max Adaptive Scale:** `{max_adaptive_scale}`
- **Perceptual Loss Type:** `{perceptual_loss_type}`
- **Number of Epochs:** `{num_epochs}`
- **Batch Size:** `{batch_size}`
- **Test Split Ratio:** `{test_split_ratio}`
- **Random Seed:** `{random_seed}`
- **Early Stopping Patience:** `{early_stopping_patience}`
- **Early Stopping Min Delta:** `{early_stopping_min_delta}`
- **Device:** `{device}`
- **Test Mode:** `{is_test_mode}`
"""
    if is_test_mode:
        readme_content += f"- **Test Samples Used:** `{test_samples}`\n"
    readme_path = output_dir / settings.README_NAME
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    logger.info(f"Saved configuration README to: {readme_path}")

    # --- Create train/test datasets and dataloaders ---
    start_time = time.time()
    
    # Get train/test file paths using data_utils
    train_paths, test_paths = get_train_test_filepaths(train_dir, test_split_ratio=test_split_ratio, random_seed=random_seed)
    
    if len(train_paths) == 0:
        logger.error("No audio files found or loaded successfully. Exiting training.")
        return
    
    # Limit samples if in test mode
    if is_test_mode and test_samples:
        train_limit = max(1, int(test_samples * (1 - test_split_ratio)))
        test_limit = max(1, test_samples - train_limit)
        
        if len(train_paths) > train_limit:
            train_paths = train_paths[:train_limit]
        if len(test_paths) > test_limit:
            test_paths = test_paths[:test_limit]
            
        logger.info(f"Test mode: limited to {len(train_paths)} training samples and {len(test_paths)} test samples")
    
    # Create datasets
    train_dataset = AudioDataset(train_paths, base_dir=train_dir)
    test_dataset = AudioDataset(test_paths, base_dir=train_dir)
    
    logger.info(f"Created datasets with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  collate_fn=collate_fn_skip_none)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn_skip_none)
    
    logger.info(f"Dataset preparation took {time.time() - start_time:.2f} seconds")

    # --- Initialize Models ---
    # Pass MESSAGE_LENGTH from settings to ensure consistency
    message_length = getattr(settings, 'MESSAGE_LENGTH', 64)  # Default to 64 if not set
    
    # Initialize with proper message length and settings
    encoder = MothEncoder(message_length=message_length, alpha=alpha, max_adaptive_scale=max_adaptive_scale).to(device)
    decoder = BatDecoder(detection_threshold=settings.DETECTION_THRESHOLD).to(device)

    # Log model parameter counts
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info(f"Encoder parameters: {encoder_params:,}")
    logger.info(f"Decoder parameters: {decoder_params:,}")

    # --- Initialize Optimizers ---
    # Use separate optimizers for encoder and decoder
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # Use separate optimizer wrapper
    optimizer = {
        'encoder': encoder_optimizer,
        'decoder': decoder_optimizer
    }
    
    # Training configuration
    config = {
        'alpha': alpha,
        'adaptive_scale': 0.001,  # Start with a small scale for stability
        'max_adaptive_scale': max_adaptive_scale,
        'perceptual_loss_type': perceptual_loss_type
    }
    
    # Add gradient clipping to prevent numerical instability
    encoder_max_grad_norm = 1.0
    decoder_max_grad_norm = 1.0
    
    # Use learning rate scheduler for better convergence
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # --- Early Stopping Initialization ---
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_encoder_state = None
    best_decoder_state = None
    stopped_early = False
    best_epoch = -1 # Track the best epoch

    # --- Training Loop ---
    epoch_times = []
    results_data = [] # Store epoch results for CSV
    logger.info(f"Starting training loop for up to {num_epochs} epochs...")
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = train_epoch(train_dataloader, encoder, decoder, optimizer, device, config)
            
            # Check if training had too many NaN batches
            nan_ratio = train_metrics.get('nan_batches', 0) / len(train_dataloader)
            if nan_ratio > 0.5:  # If more than 50% of batches had NaN
                logger.warning(f"Too many NaN batches ({train_metrics['nan_batches']}/{len(train_dataloader)}). "
                             f"Reducing max_adaptive_scale and learning rate.")
                # Reduce scale and learning rate
                config['max_adaptive_scale'] *= 0.5
                for opt in optimizer.values():
                    for param_group in opt.param_groups:
                        param_group['lr'] *= 0.5
                logger.info(f"New max_adaptive_scale: {config['max_adaptive_scale']}, "
                         f"New LR: {optimizer['encoder'].param_groups[0]['lr']}")
            
            # Evaluate on test set
            test_metrics = test_model(encoder, decoder, test_dataloader, device, perceptual_loss_type)
            
            # Log epoch summary
            epoch_duration = time.time() - epoch_start_time
            epoch_times.append(epoch_duration)
            
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            logger.info(f"- Avg Train Encoder Loss: {train_metrics['encoder_loss']:.4f}")
            logger.info(f"- Avg Train Decoder Loss: {train_metrics['decoder_loss']:.4f}")
            logger.info(f"- Avg Train Decoder Acc: {train_metrics['decoder_accuracy']:.4f}")
            logger.info(f"- Avg Train Perceptual Loss: {train_metrics['perceptual_loss']:.4f}")
            logger.info(f"- Avg Train Detection Loss: {train_metrics['detection_loss']:.4f}")
            logger.info(f"- Avg Train Scale Factor: {train_metrics['scale_factor']:.4f}")
            logger.info(f"- Avg Test Encoder Loss: {test_metrics['test_encoder_loss']:.4f}")
            logger.info(f"- Avg Test Decoder Loss: {test_metrics['test_decoder_loss']:.4f}")
            logger.info(f"- Avg Test Decoder Acc: {test_metrics['test_decoder_accuracy']:.4f}")
            logger.info(f"- Avg Test Perceptual Loss: {test_metrics['test_perceptual_loss']:.4f}")
            logger.info(f"- Avg Test Detection Loss: {test_metrics['test_detection_loss']:.4f}")
            logger.info(f"- Avg Test SNR: {test_metrics['test_snr']:.2f} dB")
            logger.info(f"- Avg Test Scale Factor: {test_metrics['avg_test_scale_factor']:.4f}")
            logger.info(f"- Epoch Time: {str(datetime.timedelta(seconds=int(epoch_duration)))}")
            
            # Estimate remaining time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            est_remaining_time = avg_epoch_time * remaining_epochs
            logger.info(f"- Est. Remaining Time: {str(datetime.timedelta(seconds=int(est_remaining_time)))}")

            # Store results for CSV
            epoch_results = {
                'epoch': epoch + 1,
                'train_encoder_loss': train_metrics['encoder_loss'],
                'train_decoder_loss': train_metrics['decoder_loss'],
                'train_decoder_acc': train_metrics['decoder_accuracy'],
                'train_perceptual_loss': train_metrics['perceptual_loss'],
                'train_detection_loss': train_metrics['detection_loss'],
                'train_scale_factor': train_metrics['scale_factor'],
                'nan_batches': train_metrics.get('nan_batches', 0),
                **{k: v for k, v in test_metrics.items()}
            }
            results_data.append(epoch_results)
            
            # --- Early Stopping Check ---
            current_test_loss = test_metrics['test_decoder_loss']
            if current_test_loss < best_test_loss - early_stopping_min_delta:
                logger.info(f"---> New best test loss found: {current_test_loss:.4f}. Saving model state.")
                best_test_loss = current_test_loss
                epochs_no_improve = 0
                # Save the best model state
                best_encoder_state = encoder.state_dict()
                best_decoder_state = decoder.state_dict()
                best_epoch = epoch + 1
                
                # Save checkpoint immediately in case training is interrupted
                encoder_save_path = output_dir / settings.ENCODER_MODEL_NAME
                decoder_save_path = output_dir / settings.DECODER_MODEL_NAME
                torch.save(best_encoder_state, encoder_save_path)
                torch.save(best_decoder_state, decoder_save_path)
                logger.info(f"Saved current best model to: {encoder_save_path}")
            else:
                epochs_no_improve += 1
                logger.info(f"---> Test loss did not improve for {epochs_no_improve} epoch(s). Best loss: {best_test_loss:.4f}")
                if epochs_no_improve >= early_stopping_patience:
                    logger.warning(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in test decoder loss for {early_stopping_patience} epochs.")
                    stopped_early = True
                    break # Exit training loop
                    
            # Update learning rate schedulers
            encoder_scheduler.step(test_metrics['test_encoder_loss'])
            decoder_scheduler.step(current_test_loss)
    
    except Exception as e:
        logger.error(f"Training interrupted by exception: {str(e)}")
        # Try to save the current best model if available
        if best_encoder_state and best_decoder_state:
            encoder_save_path = output_dir / settings.ENCODER_MODEL_NAME
            decoder_save_path = output_dir / settings.DECODER_MODEL_NAME
            torch.save(best_encoder_state, encoder_save_path)
            torch.save(best_decoder_state, decoder_save_path)
            logger.info(f"Saved best model so far to: {encoder_save_path}")

    # --- End of Training ---
    total_training_time = sum(epoch_times)
    logger.info(f"Total training time: {str(datetime.timedelta(seconds=int(total_training_time)))}")

    # Save the best models if they exist
    if best_encoder_state and best_decoder_state:
        logger.info(f"Saving model state from epoch {best_epoch} with best test loss: {best_test_loss:.4f}")
        encoder_save_path = output_dir / settings.ENCODER_MODEL_NAME
        decoder_save_path = output_dir / settings.DECODER_MODEL_NAME
        torch.save(best_encoder_state, encoder_save_path)
        torch.save(best_decoder_state, decoder_save_path)
        logger.info(f"Saved best encoder model to: {encoder_save_path}")
        logger.info(f"Saved best decoder model to: {decoder_save_path}")

        # Reload best models for final evaluation
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)
        
        # Final evaluation on test set using the best model state
        logger.info("Performing final evaluation on test set using the best saved model state...")
        final_test_metrics = test_model(encoder, decoder, test_dataloader, device, perceptual_loss_type)
        logger.info("Final Test Metrics (Best Model):")
        for key, value in final_test_metrics.items():
            if isinstance(value, float):
                logger.info(f"- {key}: {value:.4f}" + (" dB" if key == 'test_snr' else ""))
            else:
                 logger.info(f"- {key}: {value}") # Should mostly be floats

        # Add final best metrics to results data
        final_results_entry = {'epoch': 'best_final', **final_test_metrics}
        results_data.append(final_results_entry)

    else:
        logger.warning("No best model state was saved (possibly due to no improvement or error). Final evaluation skipped.")

    # Save results to CSV
    results_csv_path = output_dir / settings.RESULTS_CSV_NAME
    create_results_csv(results_data, results_csv_path)
    logger.info(f"Saved training results to: {results_csv_path}")
    
    return results_data

if __name__ == "__main__":
    """
    Main entry point when script is run directly.
    Parses command line arguments and starts training.
    """
    parser = argparse.ArgumentParser(description='Train audio steganography models with specified hyperparameters.')
    
    # Required arguments
    parser.add_argument('--train_dir', type=str, 
                        default=os.path.join('50_speakers_audio_data'),
                        help='Path to training data directory')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory name to save models (will be created in trained_models directory)')
    
    # Optional hyperparameters with defaults
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for detection loss vs perceptual loss (0-1)')
    
    parser.add_argument('--max_adaptive_scale', type=float, default=0.001,
                        help='Maximum scale for adaptive perturbation')
    
    parser.add_argument('--perceptual_loss_type', type=str, default='mse',
                        choices=['mse', 'spec', 'mel', 'psych'],
                        help='Type of perceptual loss to use')
    
    parser.add_argument('--batch_size', type=int, default=settings.BATCH_SIZE,
                        help='Training batch size')
    
    parser.add_argument('--num_epochs', type=int, default=settings.NUM_EPOCHS,
                        help='Number of training epochs')
    
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu, or None for auto-detect)')
    
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (fewer samples and epochs)')
    
    parser.add_argument('--test_samples', type=int, default=50,
                        help='Number of samples to use in test mode')
    
    parser.add_argument('--test_split_ratio', type=float, default=0.1,
                        help='Ratio of data to use for testing')
    
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of epochs with no improvement before stopping')
    
    args = parser.parse_args()
    
    # Convert output_dir to Path object
    output_dir = Path(settings.TRAINED_MODELS_DIR) / args.output_dir
    
    # Start training
    train_models(
        train_dir=args.train_dir,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        max_adaptive_scale=args.max_adaptive_scale,
        perceptual_loss_type=args.perceptual_loss_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device_str=args.device,
        is_test_mode=args.test,
        test_samples=args.test_samples,
        test_split_ratio=args.test_split_ratio,
        random_seed=args.random_seed,
        early_stopping_patience=args.early_stopping_patience
    )