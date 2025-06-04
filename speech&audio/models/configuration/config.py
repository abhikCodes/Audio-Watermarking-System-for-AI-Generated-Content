from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
from enum import Enum


class LossFunctionType(str, Enum):
    SPECTROGRAM = "spectrogram"
    LOG_MEL = "log_mel"
    PSYCHOACOUSTIC = "psychoacoustic"


class Settings(BaseSettings):
    # --- Encoder settings ---
    INPUT_SIZE: int = 96000         # Number of samples per audio segment (6s @ 16kHz)
    HIDDEN_SIZE: int = 512          # Hidden layer dimension
    CONV_KERNEL_SIZE: int = 8       # Kernel size for Conv1D layers
    CONV_STRIDE: int = 4            # Stride for Conv1D layers
    PERTURB_SCALE: float = 0.01     # Increased scale for better perceptual loss during training
    ALPHA: float = 0.1              # Weight for detection loss in encoder

    # @Yash, reduce the FFT size to 1024, if training is taking too long or increase hop_len to 1024
    # @Yash, also we can decrease with PERTURB_SCALE and ALPHA to get clearer audio. (will have to balance it, since smaller perturbations can mess up detection)
    FFT_SIZE: int = 1024            # Size of FFT (kept smaller for faster processing)
    HOP_LEN: int = 512              # Hop length for STFT
    WIN_LEN: int = 1024             # Window length for STFT (matched to FFT size)

    # --- Decoder settings ---
    DECODER_INPUT_SIZE: int = 96000
    DECODER_HIDDEN_SIZE: int = 512

    # --- Dataset settings ---
    SAMPLE_RATE: int = 16000        # Audio sampling rate
    MAX_LENGTH: int = 96000         # Max samples per segment
    NOISE_THRESHOLD: float = 0.01   # Threshold for noise gate

    # --- Training settings ---
    NUM_EPOCHS: int = 10            # Back to original
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 0.0001   # Even lower learning rate for stability
    ESTIMATE_TIME_PER_BATCH: float = 0.5  # Rough seconds per batch
    TRAIN_LOG_PATH: Path = Path("training") / "training.log"
    MODELS_DIR: Path = Path("models")
    LOSS_FUNCTION: LossFunctionType = LossFunctionType.SPECTROGRAM

    # --- Testing settings ---
    ENCODER_MODEL_PATH: Path = MODELS_DIR / "moth" / "moth_model.pth"
    DECODER_MODEL_PATH: Path = MODELS_DIR / "bat" / "bat_model.pth"
    OUTPUT_DIR: Path = Path("output")
    TEST_LOG_PATH: Path = Path("training") / "testing.log"
    PESQ_MODE: str = "wb"           # pesq narrowband vs wideband

    # --- Device settings ---
    DEVICE: Optional[str] = None  # 'cuda', 'mps', or 'cpu'; auto-detect if None

    class Config:
        # Enable loading overrides from a .env file
        env_file = ".env"
        case_sensitive = True


# Instantiate a single global settings object
settings = Settings()
