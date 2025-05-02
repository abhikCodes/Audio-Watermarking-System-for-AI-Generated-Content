from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import torch
import numpy as np
import soundfile as sf
import tempfile
import uuid
from pathlib import Path
import sys

# Add project root to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Get absolute path to project root for model loading
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Import models
from models.moth.encoder import MothEncoder
from models.bat.decoder import BatDecoder

app = FastAPI(title="Audio Steganography API", 
              description="API for processing audio clips with steganography models")

# CORS middleware setup with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Create directories for storing uploads and processed files
UPLOAD_DIR = Path(PROJECT_ROOT) / 'uploads'
PROCESSED_DIR = Path(PROJECT_ROOT) / 'processed'
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Model folders per loss
LOSS_DIRS = {
    'mse': 'mse',
    'spectrogram': 'spectrogram',
    'log_mel': 'log_mel',
    'psychoacoustic': 'psychoacoustic',
}


def load_models(loss_fn: str):
    if loss_fn not in LOSS_DIRS:
        raise HTTPException(status_code=400, detail=f"Unsupported loss function '{loss_fn}'")
    
    # Load models based on the loss function
    base = Path(PROJECT_ROOT) / 'final_models' / LOSS_DIRS[loss_fn]

    enc_path = base / 'moth' / 'moth_model.pth'
    dec_path = base / 'bat' / 'bat_model.pth'

    if not enc_path.exists() or not dec_path.exists():
        raise HTTPException(status_code=500, detail=f"Model for '{loss_fn}' not found")
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MothEncoder().to(device)
    decoder = BatDecoder().to(device)

    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder, device


@app.get("/")
async def root():
    return {
        "message": "Audio Steganography API is running", 
        "supported_loss_functions": list(LOSS_DIRS.keys())
    }

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...), loss_function: str = 'mse'):
    """
    Process an audio file with the Moth Encoder model.
    Returns the processed audio file path.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")
    
    encoder, _, device = load_models(loss_function)

    # Save uploaded file
    file_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(temp_path, 'wb') as buf: 
        shutil.copyfileobj(file.file, buf)

    
    # Process the audio file
    try:
        # Load audio
        audio, sr = sf.read(temp_path)

        if audio.ndim > 1: 
            audio = np.mean(audio, axis=1)
        
        # Ensure length is appropriate (trim or pad)
        target_length = 96000  # 6 seconds at 16kHz
        if len(audio) > target_length: 
            audio = audio[:target_length] 
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        
        # Prepare for model
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Process with Moth encoder
        with torch.no_grad():
            perturbation = encoder(audio_tensor)
            watermarked_audio = audio_tensor + perturbation
        
        # Convert back to numpy and save
        watermarked_audio = watermarked_audio.squeeze().cpu().numpy()
        
        # Save processed audio
        processed_path = PROCESSED_DIR / f"watermarked_{file_id}_{file.filename}"
        sf.write(processed_path, watermarked_audio, sr)
        
        return {
            "message": "Audio processed successfully",
            "file_id": file_id,
            "original_filename": file.filename,
            "processed_filename": f"watermarked_{file_id}_{file.filename}",
            "loss_function": loss_function
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/download-processed/{file_id}/{filename}")
async def download_processed(file_id: str, filename: str):
    """
    Download a processed audio file by ID and filename.
    """
    file_path = PROCESSED_DIR / f"watermarked_{file_id}_{filename}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed file not found")
    
    return FileResponse(path=file_path, filename=f"watermarked_{filename}", media_type="audio/wav")

@app.post("/detect-watermark/")
async def detect_watermark(file: UploadFile = File(...), loss_function: str = 'mse'):
    """
    Detect if an audio file has a watermark using the Bat Decoder.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")
    
    _, decoder, device = load_models(loss_function)

    # Save uploaded file
    file_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the audio file
    try:
        # Load audio
        audio, sr = sf.read(temp_path)
        
        # Ensure audio is mono
        if audio.ndim > 1: 
            audio = np.mean(audio, axis=1)
        
        # Ensure length is appropriate (trim or pad)
        target_length = 96000  # 6 seconds at 16kHz
        if len(audio) > target_length: 
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        
        # Prepare for model
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Detect with Bat decoder
        with torch.no_grad():
            detection = decoder(audio_tensor)
        
        watermark_probability = float(detection.item())
        
        return {
            "message": "Watermark detection completed",
            "file_id": file_id,
            "original_filename": file.filename,
            "watermark_probability": watermark_probability,
            "is_watermarked": watermark_probability > 0.5,
            "loss_function": loss_function
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting watermark: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
