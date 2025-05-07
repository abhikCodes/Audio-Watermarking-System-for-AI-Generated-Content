import torch
import librosa
import numpy as np
import os
import sys
import logging
import soundfile as sf
from pesq import pesq
from tqdm import tqdm
from configuration.config import settings

sys.path.append('..')

from models.moth.encoder import MothEncoder
from models.bat.decoder import BatDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training/testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SteganographyTesting")

# --- Auto-detect device ---
## @Abhik this is done so that we can run the code better for apple devices change this to cuda if you do google collab also i would want this to be configurable inine function can you make thta happen?
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

def load_audio(audio_path):
    """Load and preprocess audio file"""
    logger.info(f"Loading audio file: {audio_path}")

    sample_rate = settings.SAMPLE_RATE
    max_length = settings.MAX_LENGTH
    
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Pad or truncate
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    else:
        audio = audio[:max_length]
    
    return audio

def calculate_pesq_score(original, processed):
    """Calculate PESQ score between original and processed audio"""
    try:
        sample_rate = settings.SAMPLE_RATE
        score = pesq(sample_rate, original, processed, settings.PESQ_MODE)
        return score
    except Exception as e:
        logger.error(f"Error calculating PESQ score: {e}")
        return 0.0

def test_models(encoder_path, decoder_path, test_dir, output_dir):
    """
    Test the trained encoder and decoder models
    Args:
        encoder_path: Path to the trained encoder model
        decoder_path: Path to the trained decoder model
        test_dir: Directory containing test audio files
        output_dir: Directory to save watermarked audio
        device_str: Device to run inference on ('cuda', 'mps', or 'cpu'). Auto-detects if None.
    """
    # --- Determine device ---
    device_str = settings.DEVICE
    if device_str is None:
        device_str = get_default_device()
    
    try:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    except RuntimeError as e:
        logger.error(f"Could not use device '{device_str}'. Error: {e}. Falling back to CPU.")
        device = torch.device('cpu')
    # --------------------------

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    logger.info("Loading trained models...")
    encoder = MothEncoder().to(device)
    decoder = BatDecoder().to(device)
    
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    
    encoder.eval()
    decoder.eval()
    
    # Find test files
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.wav'):
                test_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(test_files)} test files")
    
    # Test metrics
    total_files = len(test_files)
    successful_detections = 0
    false_positives = 0
    avg_pesq = 0.0
    
    # Process each test file
    for file_path in tqdm(test_files, desc="Testing"):
        # Load and preprocess audio
        audio = load_audio(file_path)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Test original audio with decoder (should be classified as 0)
        with torch.no_grad():
            original_pred = decoder(audio_tensor)
            original_pred_np = original_pred.cpu().numpy()[0][0]
        
        # Generate watermarked audio
        with torch.no_grad():
            perturbation = encoder(audio_tensor)
            watermarked = audio_tensor + perturbation
            watermarked_np = watermarked.cpu().numpy()[0][0]
        
        # Test watermarked audio with decoder (should be classified as 1)
        with torch.no_grad():
            watermarked_pred = decoder(watermarked)
            watermarked_pred_np = watermarked_pred.cpu().numpy()[0][0]
        
        # Calculate PESQ score
        pesq_score = calculate_pesq_score(audio, watermarked_np)
        avg_pesq += pesq_score
        
        # Save watermarked audio
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.wav', '_watermarked.wav'))
        sf.write(output_path, watermarked_np, settings.SAMPLE_RATE)
        
        # Update metrics
        if watermarked_pred_np > 0.5:
            successful_detections += 1
        if original_pred_np > 0.5:
            false_positives += 1
        
        # Log results for this file
        logger.info(f"File: {os.path.basename(file_path)}")
        logger.info(f"  Original prediction: {original_pred_np:.4f}")
        logger.info(f"  Watermarked prediction: {watermarked_pred_np:.4f}")
        logger.info(f"  PESQ score: {pesq_score:.4f}")
    
    # Calculate final metrics
    detection_accuracy = successful_detections / total_files
    false_positive_rate = false_positives / total_files
    avg_pesq = avg_pesq / total_files
    
    # Log summary
    logger.info("\nTest Results Summary:")
    logger.info(f"Total test files: {total_files}")
    logger.info(f"Detection accuracy: {detection_accuracy:.4f}")
    logger.info(f"False positive rate: {false_positive_rate:.4f}")
    logger.info(f"Average PESQ score: {avg_pesq:.4f}")
    
    return {
        'detection_accuracy': detection_accuracy,
        'false_positive_rate': false_positive_rate,
        'avg_pesq': avg_pesq
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test Moth and Bat models')
    parser.add_argument('--encoder_path', type=str, default=settings.ENCODER_MODEL_PATH, help='Path to the trained encoder model')
    parser.add_argument('--decoder_path', type=str, default=settings.DECODER_MODEL_PATH, help='Path to the trained decoder model')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test audio files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save watermarked audio')
    # parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., cuda, mps, cpu). Auto-detects if not specified.')
    
    args = parser.parse_args()
    # test_models(args.encoder_path, args.decoder_path, args.test_dir, args.output_dir, args.device) 
    test_models(args.encoder_path, args.decoder_path, args.test_dir, args.output_dir) 