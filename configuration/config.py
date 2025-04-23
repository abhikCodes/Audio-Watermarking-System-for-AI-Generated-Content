from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # --- Encoder settings ---
    #@Yash, also we can decrease with PERTURB_SCALE and ALPHA to get clearer audio. (will have to balance it, since smaller perturbations can mess up detection)
    INPUT_SIZE: int = 96000         # Number of samples per audio segment (6s @ 16kHz)
    HIDDEN_SIZE: int = 512          # Hidden layer dimension
    CONV_KERNEL_SIZE: int = 8       # Kernel size for Conv1D layers
    CONV_STRIDE: int = 4            # Stride for Conv1D layers
    PERTURB_SCALE: float = 0.001    # Scale for output of encoder tanh
    ALPHA: float = 0.07             # Weight for detection loss in encoder

    # @Yash, reduce the FFT size to 1024, if training is taking too long or increase hop_len to 1024
    FFT_SIZE: int = 2048           # Size of FFT
    HOP_LEN: int = 512              # Hop length for STFT
    WIN_LEN: int = 2048             # Window length for STFT

    # --- Decoder settings ---
    DECODER_INPUT_SIZE: int = 96000
    DECODER_HIDDEN_SIZE: int = 512

    # --- Dataset settings ---
    SAMPLE_RATE: int = 15000       # Audio sampling rate
    MAX_LENGTH: int = 96000         # Max samples per segment
    NOISE_THRESHOLD: float = 0.01   # Threshold for noise gate

    # --- Training settings ---
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 0.001
    ESTIMATE_TIME_PER_BATCH: float = 0.5  # Rough seconds per batch
    TRAIN_LOG_PATH: Path = Path("training") / "training.log"
    MODELS_DIR: Path = Path("models")

    # --- Testing settings ---
    ENCODER_MODEL_PATH: Path = MODELS_DIR / "moth" / "moth_model.pth"
    DECODER_MODEL_PATH: Path = MODELS_DIR / "bat" / "bat_model.pth"
    OUTPUT_DIR: Path = Path("output")
    TEST_LOG_PATH: Path = Path("training") / "testing.log"
    PESQ_MODE: str = "wb"           # pesq narrowband vs wideband

    # --- Device settings ---
    DEVICE: Optional[str] = 'mps'  # 'cuda', 'mps', or 'cpu'; auto-detect if None

    class Config:
        # Enable loading overrides from a .env file
        env_file = ".env"


# Instantiate a single global settings object
settings = Settings()
