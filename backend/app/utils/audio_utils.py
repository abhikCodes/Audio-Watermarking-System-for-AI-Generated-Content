import torch
import torchaudio
import os
from typing import Tuple

def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load and preprocess an audio file.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    wav, sr = torchaudio.load(file_path)
    wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
    wav = wav / (wav.abs().max() + 1e-9)  # Normalize
    return wav, sample_rate

def save_audio(audio: torch.Tensor, file_path: str, sample_rate: int = 16000):
    """
    Save an audio tensor to a file.
    
    Args:
        audio: Audio tensor
        file_path: Path to save the audio file
        sample_rate: Sample rate of the audio
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Convert to int16 format for WAV
    arr = audio.squeeze().cpu().numpy()
    arr = (arr * 32767).astype('int16')
    torchaudio.save(file_path, torch.from_numpy(arr).unsqueeze(0), sample_rate)

def process_audio_for_model(audio: torch.Tensor, target_length: int = None) -> torch.Tensor:
    """
    Process audio to match model input requirements. If target_length is None, return the full audio.
    """
    if target_length is not None:
        L = audio.shape[-1]
        if L >= target_length:
            audio = audio[:, :target_length]
        else:
            repeats = (target_length + L - 1) // L
            audio = audio.repeat(1, repeats)[:, :target_length]
    return audio 