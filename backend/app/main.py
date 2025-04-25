from fastapi import FastAPI, UploadFile, File, HTTPException
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
    allow_origins="*",  # Frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Create directories for storing uploads and processed files
UPLOAD_DIR = Path("../uploads")
PROCESSED_DIR = Path("../processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
moth_encoder = MothEncoder().to(device)
bat_decoder = BatDecoder().to(device)

# Model paths using absolute paths based on project root
ENCODER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/moth/moth_model.pth')
DECODER_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/bat/bat_model.pth')

# Try to load model weights if available
try:
    if os.path.exists(ENCODER_MODEL_PATH):
        moth_encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
        moth_encoder.eval()
        print(f"Loaded Moth Encoder weights from {ENCODER_MODEL_PATH}")
    else:
        print(f"Warning: Model file {ENCODER_MODEL_PATH} does not exist")
        print("Using untrained Moth Encoder - this is expected during development but won't provide optimal results")
except Exception as e:
    print(f"Could not load Moth Encoder weights: {e}")
    print("Using untrained Moth Encoder - functionality will be limited")

try:
    if os.path.exists(DECODER_MODEL_PATH):
        bat_decoder.load_state_dict(torch.load(DECODER_MODEL_PATH, map_location=device))
        bat_decoder.eval()
        print(f"Loaded Bat Decoder weights from {DECODER_MODEL_PATH}")
    else:
        print(f"Warning: Model file {DECODER_MODEL_PATH} does not exist")
        print("Using untrained Bat Decoder - this is expected during development but won't provide optimal results")
except Exception as e:
    print(f"Could not load Bat Decoder weights: {e}")
    print("Using untrained Bat Decoder - functionality will be limited")

@app.get("/")
async def root():
    return {"message": "Audio Steganography API is running"}

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    Process an audio file with the Moth Encoder model.
    Returns the processed audio file path.
    """
    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")
    
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
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure length is appropriate (trim or pad)
        target_length = 96000  # 6 seconds at 16kHz
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Prepare for model
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Process with Moth encoder
        with torch.no_grad():
            perturbation = moth_encoder(audio_tensor)
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
            "processed_filename": f"watermarked_{file_id}_{file.filename}"
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
async def detect_watermark(file: UploadFile = File(...)):
    """
    Detect if an audio file has a watermark using the Bat Decoder.
    """
    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")
    
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
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure length is appropriate (trim or pad)
        target_length = 96000  # 6 seconds at 16kHz
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Prepare for model
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Detect with Bat decoder
        with torch.no_grad():
            detection = bat_decoder(audio_tensor)
        
        watermark_probability = float(detection.item())
        
        return {
            "message": "Watermark detection completed",
            "file_id": file_id,
            "original_filename": file.filename,
            "watermark_probability": watermark_probability,
            "is_watermarked": watermark_probability > 0.5
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting watermark: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)