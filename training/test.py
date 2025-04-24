import torch
import librosa
import numpy as np
import os
import sys
import logging
import soundfile as sf
from pesq import pesq
from tqdm import tqdm
import argparse
from pathlib import Path
import re
import csv
from typing import Optional, Dict, Any
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoder import MothEncoder
from models.decoder import BatDecoder
from configuration.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.TEST_LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SteganographyTesting")

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

def parse_readme(readme_path: Path) -> Dict[str, Any]:
    """Parse hyperparameters from the README.md file in the model directory."""
    hyperparams = {
        'alpha': None,
        'perturb_scale': None,
        # Add others if needed, e.g., learning_rate, perceptual_loss_type
    }
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
            # Simple parsing based on expected format
            alpha_match = re.search(r"- **Alpha \(Detection Loss Weight\):\*\*\s*`?([\d.]+)`?\"", content)
            perturb_match = re.search(r"- **Perturbation Scale:\*\*\s*`?([\d.]+)`?\"", content)

            if alpha_match:
                hyperparams['alpha'] = float(alpha_match.group(1))
            if perturb_match:
                hyperparams['perturb_scale'] = float(perturb_match.group(1))

            if hyperparams['alpha'] is None or hyperparams['perturb_scale'] is None:
                logger.warning(f"Could not parse all hyperparameters from {readme_path}")

    except FileNotFoundError:
        logger.error(f"README file not found at {readme_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing README {readme_path}: {e}")
        return None

    return hyperparams

# Simple Dataset for testing (similar to training one, but only loads paths)
class AudioTestDataset(Dataset):
    def __init__(self, audio_dir, max_samples=None):
        self.audio_paths = []
        self.max_length = settings.MAX_LENGTH
        self.sample_rate = settings.SAMPLE_RATE
        potential_paths = []
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    potential_paths.append(os.path.join(root, file))

        if max_samples is not None and len(potential_paths) > max_samples:
            logger.info(f"Limiting test dataset to {max_samples} samples.")
            self.audio_paths = potential_paths[:max_samples]
        else:
            self.audio_paths = potential_paths
        logger.info(f"Found {len(self.audio_paths)} WAV files for testing.")

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            if len(audio) < self.max_length:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
            else:
                audio = audio[:self.max_length]
            # Return path along with audio numpy array
            return os.path.basename(audio_path), audio
        except Exception as e:
            logger.warning(f"Could not load test audio file: {audio_path}. Error: {e}")
            return None, None # Indicate failure

def collate_test_fn(batch):
    # Filter out None entries (where loading failed)
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    # Separate paths and audio arrays
    paths = [item[0] for item in batch]
    audio_arrays = [item[1] for item in batch]
    # Collate audio arrays into a tensor
    audio_tensors = default_collate([torch.from_numpy(a).float().unsqueeze(0) for a in audio_arrays])
    return paths, audio_tensors

def calculate_pesq_score(original, processed, sample_rate):
    """Calculate PESQ score between original and processed audio"""
    try:
        # Ensure inputs are numpy float arrays
        original_np = np.asarray(original, dtype=np.float32)
        processed_np = np.asarray(processed, dtype=np.float32)
        score = pesq(sample_rate, original_np, processed_np, settings.PESQ_MODE)
        return score
    except Exception as e:
        logger.error(f"Error calculating PESQ score: {e}")
        # Potentially return NaN or raise specific error
        return np.nan

def test_models(
    model_dir: Path,
    test_dir: str,
    device_str: Optional[str] = None,
    is_test_mode: bool = False,
    test_samples: Optional[int] = None
):
    """
    Test trained models from a specific run directory.
    Loads hyperparameters from README, evaluates on test_dir, saves results to CSV.
    """
    logger.info(f"--- Starting Test Run for Model Directory: {model_dir} ---")

    # --- Parse README --- 
    readme_path = model_dir / settings.README_NAME
    hyperparams = parse_readme(readme_path)
    if hyperparams is None or hyperparams.get('alpha') is None or hyperparams.get('perturb_scale') is None:
        logger.error("Failed to load hyperparameters from README. Aborting test.")
        return
    logger.info(f"Loaded hyperparameters: Alpha={hyperparams['alpha']}, PerturbScale={hyperparams['perturb_scale']}")

    # --- Determine Device ---
    if device_str is None:
        device_str = get_default_device()
    try:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    except RuntimeError as e:
        logger.error(f"Could not use device '{device_str}'. Error: {e}. Falling back to CPU.")
        device = torch.device('cpu')

    # --- Load Models ---
    encoder_path = model_dir / settings.ENCODER_MODEL_NAME
    decoder_path = model_dir / settings.DECODER_MODEL_NAME

    if not encoder_path.exists() or not decoder_path.exists():
        logger.error(f"Model files not found in {model_dir}. Expected {encoder_path} and {decoder_path}")
        return

    logger.info("Loading trained models...")
    try:
        encoder = MothEncoder(alpha=hyperparams['alpha'], perturb_scale=hyperparams['perturb_scale']).to(device)
        decoder = BatDecoder(detection_threshold=settings.DETECTION_THRESHOLD).to(device)

        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))

        encoder.eval()
        decoder.eval()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return

    # --- Create Dataset & Dataloader ---
    max_ds_samples = test_samples if is_test_mode else None
    dataset = AudioTestDataset(test_dir, max_samples=max_ds_samples)
    if len(dataset) == 0:
        logger.error("No test audio files found or loaded.")
        return
    # Use a batch size of 1 for testing to process file-by-file easily
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_test_fn)

    # --- Evaluation Loop ---
    results_list = []
    logger.info(f"Starting evaluation on {len(dataset)} files...")

    for file_paths, audio_tensors in tqdm(dataloader, desc="Evaluating"):
        if file_paths is None or audio_tensors is None:
            continue # Skip batches where all files failed to load

        # Since batch_size=1, get the single item
        file_path = file_paths[0]
        audio_tensor = audio_tensors.to(device) # Shape: [1, 1, num_samples]
        # Get original numpy audio array (needed for PESQ)
        # We need to reload or have stored the original numpy array before tensor conversion
        # Let's reload for simplicity here, though inefficient.
        # A better dataset would yield both tensor and original numpy array.
        try:
            original_audio_np, _ = librosa.load(os.path.join(test_dir, file_path), sr=settings.SAMPLE_RATE, mono=True)
            if np.max(np.abs(original_audio_np)) > 0:
                original_audio_np = original_audio_np / np.max(np.abs(original_audio_np))
            if len(original_audio_np) < settings.MAX_LENGTH:
                original_audio_np = np.pad(original_audio_np, (0, settings.MAX_LENGTH - len(original_audio_np)))
            else:
                original_audio_np = original_audio_np[:settings.MAX_LENGTH]
        except Exception as e:
            logger.warning(f"Could not reload original numpy for PESQ for {file_path}: {e}")
            original_audio_np = None

        with torch.no_grad():
            # Original Detection
            original_logits = decoder(audio_tensor)
            original_prob = torch.sigmoid(original_logits).item()
            original_detected = (original_prob > decoder.detection_threshold)

            # Watermarked Detection
            perturbation = encoder(audio_tensor)
            watermarked_tensor = audio_tensor + perturbation
            watermarked_logits = decoder(watermarked_tensor)
            watermarked_prob = torch.sigmoid(watermarked_logits).item()
            watermarked_detected = (watermarked_prob > decoder.detection_threshold)

            # Get watermarked numpy for PESQ
            watermarked_np = watermarked_tensor.squeeze().cpu().numpy()

        # Calculate PESQ
        pesq_score = np.nan
        if original_audio_np is not None:
            pesq_score = calculate_pesq_score(original_audio_np, watermarked_np, settings.SAMPLE_RATE)

        # Store results
        results_list.append({
            'filename': file_path,
            'pesq': pesq_score,
            'original_prob': original_prob,
            'watermarked_prob': watermarked_prob,
            'original_detected': original_detected,
            'watermarked_detected': watermarked_detected
        })

    # --- Save Results to CSV ---
    if not results_list:
        logger.warning("No results were generated during testing.")
        return

    results_df = pd.DataFrame(results_list)
    csv_save_path = model_dir / settings.RESULTS_CSV_NAME
    try:
        results_df.to_csv(csv_save_path, index=False)
        logger.info(f"Saved test results to: {csv_save_path}")
    except Exception as e:
        logger.error(f"Failed to save results CSV: {e}")

    # --- Log Summary Metrics ---
    avg_pesq = results_df['pesq'].mean() # Note: mean() ignores NaNs by default
    detection_accuracy = results_df['watermarked_detected'].mean()
    false_positive_rate = results_df['original_detected'].mean()
    logger.info("\n--- Test Results Summary ---")
    logger.info(f"Total files evaluated: {len(results_df)}")
    logger.info(f"Detection Accuracy (Watermarked): {detection_accuracy:.4f}")
    logger.info(f"False Positive Rate (Original): {false_positive_rate:.4f}")
    logger.info(f"Average PESQ: {avg_pesq:.4f}")
    logger.info(f"Results saved to {csv_save_path}")
    logger.info(f"--- End Test Run for {model_dir} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained Moth and Bat models from a specific run directory.')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the directory containing trained models (encoder.pth, decoder.pth) and README.md')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test audio files')
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of audio samples to use in test mode')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., cuda, mps, cpu). Auto-detects if not specified.')

    args = parser.parse_args()

    # Construct full model path
    model_path = Path(args.model_dir) # No need to join with TRAINED_MODELS_DIR, assume full path is given

    if not model_path.is_dir():
        logger.error(f"Model directory not found: {model_path}")
        sys.exit(1)

    # Call testing function
    test_models(
        model_dir=model_path,
        test_dir=args.test_dir,
        device_str=args.device,
        is_test_mode=args.test,
        test_samples=args.test_samples
    ) 