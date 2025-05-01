import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate # Import default_collate
import numpy as np
import librosa
import os
import time
import datetime
from tqdm import tqdm
import logging
import sys
import os
import shutil
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.moth.encoder import MothEncoder
from models.bat.decoder import BatDecoder
from configuration.config import settings


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
        return 'cpu'
# --------------------------

class AudioDataset(Dataset):
    def __init__(self, audio_dir):
        """
        Dataset for audio steganography
        Args:
            audio_dir: Directory containing audio files
            max_length: Maximum length of audio segments
            sample_rate: Sample rate for audio loading
        """
        self.audio_paths = []
        self.max_length = settings.MAX_LENGTH
        self.sample_rate = settings.SAMPLE_RATE
        self.failed_loads = 0 # Keep track of failed loads
        
        # Collect all WAV files
        logger.info(f"Scanning directory {audio_dir} for WAV files...")
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    self.audio_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(self.audio_paths)} potential WAV files for training")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Preprocessing
            audio = self.normalize_audio(audio)
            
            # Convert to tensor
            audio = torch.from_numpy(audio).float()
            
            # Pad or truncate
            if len(audio) < self.max_length:
                audio = torch.nn.functional.pad(audio, (0, self.max_length - len(audio)))
            else:
                audio = audio[:self.max_length]
            
            return audio.unsqueeze(0)  # Add channel dimension
        except Exception as e:
            # Log the error and return None if loading fails
            logger.warning(f"Could not load audio file: {audio_path}. Error: {e}")
            self.failed_loads += 1
            return None # Indicate failure
    
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
    # Filter out None entries from the batch
    batch = [item for item in batch if item is not None]
    # If the batch is empty after filtering, return None or an empty tensor
    if not batch:
        return None
    # Use the default collate function on the filtered batch
    return default_collate(batch)
# -----------------------------

def train_models(train_dir, device_str=None):
    """
    Train both Moth and Bat models
    Args:
        train_dir: Directory containing training audio files
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device_str: Device to train on ('cuda', 'mps', or 'cpu'). Auto-detects if None.
    """
    num_epochs = settings.NUM_EPOCHS
    batch_size = settings.BATCH_SIZE
    # --- Determine device ---
    if device_str is None:
        device_str = get_default_device()
    
    try:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    except RuntimeError as e:
        logger.error(f"Could not use device '{device_str}'. Error: {e}. Falling back to CPU.")
        device = torch.device('cpu')
    # --------------------------

    # Log training parameters
    logger.info("Starting training with parameters:")
    logger.info(f"- Training directory: {train_dir}")
    logger.info(f"- Number of epochs: {num_epochs}")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Device: {device}")
    logger.info(f"- Loss function: {settings.LOSS_FUNCTION.value}")
    
    # Create model save directories
    loss_type = settings.LOSS_FUNCTION.value
    
    try:
        # Create only the final models directories - nothing else
        final_models_dir = Path("final_models")
        final_moth_dir = final_models_dir / loss_type / "moth"
        final_bat_dir = final_models_dir / loss_type / "bat"
        
        # Create minimum directories needed
        os.makedirs(final_moth_dir, exist_ok=True)
        os.makedirs(final_bat_dir, exist_ok=True)
        
        logger.info(f"Created model directories successfully")
    except OSError as e:
        logger.error(f"Failed to create model directories: {e}")
        logger.info("Will attempt to continue without saving models")

    # Create dataset and dataloader
    start_time = time.time()
    dataset = AudioDataset(train_dir)

    # Use the custom collate function to get rid of NULL/Corrupted entries from batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_skip_none)
    logger.info(f"Dataset preparation took {time.time() - start_time:.2f} seconds")
    
    # Estimate total training time based on potentially fewer valid samples
    # This is a rough estimate, actual time depends on how many files fail
    num_valid_samples = len(dataset) - dataset.failed_loads # Get initial failed count
    if num_valid_samples <= 0:
        logger.error("No valid audio files found for training. Exiting.")
        return
        
    batches_per_epoch = (num_valid_samples + batch_size - 1) // batch_size
    estimated_time_per_epoch = batches_per_epoch * 0.5  # Rough estimate
    estimated_total_time = estimated_time_per_epoch * num_epochs
    estimated_completion_time = datetime.datetime.now() + datetime.timedelta(seconds=estimated_total_time)

    logger.info(f"Estimated training time (approx): {str(datetime.timedelta(seconds=int(estimated_total_time)))}")
    logger.info(f"Estimated completion time (approx): {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize models
    encoder = MothEncoder().to(device)
    decoder = BatDecoder().to(device)
    
    # Log model architectures
    logger.info(f"Moth Encoder Architecture:\n{encoder}")
    logger.info(f"Bat Decoder Architecture:\n{decoder}")
    
    # Initialize optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=settings.LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=settings.LEARNING_RATE)
    
    # Initialize PESQ trackers for monitoring audio quality
    best_pesq_score = 0.0
    
    # Training loop
    epoch_times = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        encoder.train()
        decoder.train()
        
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_decoder_accuracy = 0
        total_perceptual_loss = 0
        total_detection_loss = 0
        batches_processed = 0 # Track batches actually processed
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, audio_batch in enumerate(progress_bar):
            # Check if the batch is None (all files in batch failed to load)
            if audio_batch is None:
                logger.warning(f"Skipping empty batch at index {batch_idx}")
                continue
                
            audio = audio_batch.to(device)
            batch_size_actual = audio.size(0) # Actual batch size might be smaller
            batches_processed += 1
            
            # Train Decoder
            decoder_optimizer.zero_grad()
            
            # --- Decoder: Original Audio ---
            pred_original = decoder(audio)
            loss_original, acc_original = decoder.compute_loss(
                pred_original, 
                torch.zeros(batch_size_actual, 1).to(device)
            )
            
            # --- Decoder: Watermarked Audio ---
            # Generate watermarked audio (detached from encoder graph for this step)
            with torch.no_grad():
                perturbation_dec = encoder(audio).detach()
            watermarked_dec = audio + perturbation_dec
            # Predict on watermarked audio
            pred_watermarked_dec = decoder(watermarked_dec)
            loss_watermarked, acc_watermarked = decoder.compute_loss(
                pred_watermarked_dec,
                torch.ones(batch_size_actual, 1).to(device)
            )
            
            # --- Decoder: Backward and Step ---
            decoder_loss = (loss_original + loss_watermarked) / 2
            # No need to retain graph here as we recalculate for encoder
            decoder_loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            decoder_optimizer.step()
            
            # Train Encoder
            encoder_optimizer.zero_grad()
            
            # --- Encoder: Generate Watermarked and Get Decoder Prediction ---
            # Re-generate perturbation and watermarked audio *with* encoder graph
            perturbation_enc = encoder(audio)
            watermarked_enc = audio + perturbation_enc
            # Get decoder's prediction on this *new* watermarked audio
            pred_watermarked_enc = decoder(watermarked_enc)
            
            # --- Encoder: Compute Loss, Backward and Step ---
            encoder_loss, metrics = encoder.compute_loss(audio, watermarked_enc, pred_watermarked_enc)
            encoder_loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            encoder_optimizer.step()
            
            # Skip batch if NaN values detected in loss
            if torch.isnan(torch.tensor(encoder_loss.item())) or torch.isnan(torch.tensor(decoder_loss.item())):
                logger.warning(f"NaN loss detected in batch {batch_idx}, skipping update")
                continue
            
            # Update metrics
            total_encoder_loss += metrics['total_loss']
            total_decoder_loss += decoder_loss.item()
            total_decoder_accuracy += (acc_original + acc_watermarked) / 2
            total_perceptual_loss += metrics['perceptual_loss']
            total_detection_loss += metrics['detection_loss']
            
            # Update progress bar with cleaner format
            progress_bar.set_postfix({
                'enc_loss': f'{metrics["total_loss"]:.6f}',
                'dec_loss': f'{decoder_loss.item():.6f}',
                'dec_acc': f'{((acc_original + acc_watermarked) / 2):.4f}'
            })
            
            # No per-epoch model saving to conserve disk space
        
        # Check if any batches were processed
        if batches_processed == 0:
            logger.error(f"Epoch {epoch+1}/{num_epochs}: No batches were successfully processed. Check data loading logs.")
            continue # Skip to next epoch or break

        # Calculate epoch metrics based on batches processed
        avg_encoder_loss = total_encoder_loss / batches_processed
        avg_decoder_loss = total_decoder_loss / batches_processed
        avg_decoder_accuracy = total_decoder_accuracy / batches_processed
        avg_perceptual_loss = total_perceptual_loss / batches_processed
        avg_detection_loss = total_detection_loss / batches_processed
        
        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Estimate remaining time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        # Log epoch summary
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}:")
        logger.info(f"- Batches Processed: {batches_processed}")
        logger.info(f"- Average Encoder Loss: {avg_encoder_loss:.4f}")
        logger.info(f"- Average Decoder Loss: {avg_decoder_loss:.4f}")
        logger.info(f"- Average Decoder Accuracy: {avg_decoder_accuracy:.4f}")
        logger.info(f"- Average Perceptual Loss: {avg_perceptual_loss:.4f}")
        logger.info(f"- Average Detection Loss: {avg_detection_loss:.4f}")
        logger.info(f"- Epoch Time: {str(datetime.timedelta(seconds=int(epoch_time)))}")
        logger.info(f"- Estimated Remaining Time: {str(datetime.timedelta(seconds=int(estimated_remaining_time)))}")
        
        # No per-epoch model saving to conserve disk space
    
    # After all epochs are done, save only the final models
    try:
        # Save only final models
        logger.info("Training complete, saving final models...")
        final_models_dir = Path("final_models")
        final_moth_dir = final_models_dir / loss_type / "moth"
        final_bat_dir = final_models_dir / loss_type / "bat"
        
        # Save models
        torch.save(encoder.state_dict(), final_moth_dir / 'moth_model.pth')
        torch.save(decoder.state_dict(), final_bat_dir / 'bat_model.pth')
        logger.info(f"Final models saved successfully to final_models/{loss_type}/")
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to save models: {e}")
    
    # Log total training time and failed loads
    total_training_time = time.time() - start_time
    logger.info(f"Total training time: {str(datetime.timedelta(seconds=int(total_training_time)))}")
    logger.info(f"Total files that failed to load: {dataset.failed_loads}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Moth and Bat models')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training audio files')
    # parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    # parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    # parser.add_argument('--device', type=str, default=None, help='Device to train on (e.g., cuda, mps, cpu). Auto-detects if not specified.')
    
    args = parser.parse_args()
    # train_models(args.train_dir, args.num_epochs, args.batch_size, args.device)
    train_models(args.train_dir)