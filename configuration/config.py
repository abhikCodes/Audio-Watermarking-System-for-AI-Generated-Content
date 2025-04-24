from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # --- Model structure settings (Shared for Encoder/Decoder where applicable) ---
    INPUT_SIZE: int = 96000         # Number of samples per audio segment (6s @ 16kHz)
    HIDDEN_SIZE: int = 512          # Hidden layer dimension
    CONV_KERNEL_SIZE: int = 8       # Kernel size for Conv1D layers
    CONV_STRIDE: int = 4            # Stride for Conv1D layers
    MESSAGE_LENGTH: int = 64        # Length of the binary message to be encoded

    # --- Encoder-specific Spectrogram settings ---
    # @Yash, reduce the FFT size to 1024, if training is taking too long or increase hop_len to 1024
    FFT_SIZE: int = 2048           # Size of FFT
    HOP_LEN: int = 512             # Hop length for STFT
    WIN_LEN: int = 2048            # Window length for STFT
    N_MELS: int = 128              # Number of mel bands

    # --- Decoder-specific settings ---
    DECODER_INPUT_SIZE: int = 96000
    DECODER_HIDDEN_SIZE: int = 512
    DETECTION_THRESHOLD: float = 0.5 # Threshold for watermark detection

    # --- Dataset settings ---
    SAMPLE_RATE: int = 15000       # Audio sampling rate
    MAX_LENGTH: int = 96000         # Max samples per segment
    NOISE_THRESHOLD: float = 0.01   # Threshold for noise gate

    # --- Default Training settings (Can be overridden by experiment runner) ---
    NUM_EPOCHS: int = 10 # Base epochs, can be overridden for test runs
    BATCH_SIZE: int = 4
    ESTIMATE_TIME_PER_BATCH: float = 0.5  # Rough seconds per batch

    # --- Paths and Logging ---
    TRAIN_LOG_PATH: Path = Path("training") / "training.log"
    TEST_LOG_PATH: Path = Path("training") / "testing.log"
    TRAINED_MODELS_DIR: Path = Path("trained_models")
    # Base names for files within experiment directories
    ENCODER_MODEL_NAME: str = "encoder.pth"
    DECODER_MODEL_NAME: str = "decoder.pth"
    RESULTS_CSV_NAME: str = "results.csv"
    README_NAME: str = "README.md"

    OUTPUT_DIR: Path = Path("output") # General output, might not be used
    PESQ_MODE: str = "wb"           # pesq narrowband vs wideband

    # --- Device settings ---
    DEVICE: Optional[str] = 'mps'  # 'cuda', 'mps', or 'cpu'; auto-detect if None

    # --- Loss Flags (Defaults, can be part of hyperparameter tuning if needed) ---
    # Using MSE for perceptual loss as a default example
    # Other options like 'spec', 'mel', 'psych' could be added or made tunable
    PERCEPTUAL_LOSS_TYPE: str = 'mse' # Example default

    class Config:
        # Enable loading overrides from a .env file
        env_file = ".env"


# Instantiate a single global settings object
settings = Settings()
