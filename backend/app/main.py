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
from datetime import datetime

# Import local models
from app.models.models import MothEncoder, BatDetector
from app.utils.audio_utils import load_audio, save_audio, process_audio_for_model

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
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Constants
SAMPLE_RATE = 16000
TARGET_LENGTH = 480000  # 30 seconds at 16kHz
ALPHA = 0.1  # Watermark strength
MODEL_DIR = "app/working_model"

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
moth = MothEncoder(ALPHA).to(device)
bat = BatDetector().to(device)

# Load model weights
moth.load_state_dict(torch.load(os.path.join(MODEL_DIR, "moth_final.ckpt"), map_location=device))
bat.load_state_dict(torch.load(os.path.join(MODEL_DIR, "bat_final.ckpt"), map_location=device))

moth.eval()
bat.eval()

@app.get("/")
async def root():
    return {
        "message": "Audio Steganography API is running",
        "sample_rate": SAMPLE_RATE,
        "target_length": f"{TARGET_LENGTH/SAMPLE_RATE} seconds"
    }

@app.post("/watermark")
async def watermark_audio(file: UploadFile = File(...)):
    """
    Add watermark to an audio file
    """
    try:
        if not file.filename.lower().endswith(('.wav', '.mp3')):
            raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")

        # Save uploaded file temporarily
        temp_input = f"uploads/temp_input_{datetime.now().timestamp()}.wav"
        temp_output = f"processed/watermarked_{datetime.now().timestamp()}.wav"
        
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("processed", exist_ok=True)
        
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process audio
        audio, _ = load_audio(temp_input, SAMPLE_RATE)
        audio = process_audio_for_model(audio)
        
        # Add watermark
        with torch.no_grad():
            audio = audio.unsqueeze(0).to(device)
            watermarked = moth(audio)
            watermarked = watermarked.squeeze(0).cpu()
        
        # Save watermarked audio
        save_audio(watermarked, temp_output, SAMPLE_RATE)
        
        # Clean up input file
        os.remove(temp_input)
        
        return FileResponse(
            temp_output,
            media_type="audio/wav",
            filename="watermarked.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_watermark(file: UploadFile = File(...)):
    """
    Detect watermark in an audio file
    """
    try:
        if not file.filename.lower().endswith(('.wav', '.mp3')):
            raise HTTPException(status_code=400, detail="Only .wav or .mp3 files are supported")

        # Save uploaded file temporarily
        temp_input = f"uploads/temp_input_{datetime.now().timestamp()}.wav"
        
        os.makedirs("uploads", exist_ok=True)
        
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process audio
        audio, _ = load_audio(temp_input, SAMPLE_RATE)
        audio = process_audio_for_model(audio)
        
        # Detect watermark
        with torch.no_grad():
            audio = audio.unsqueeze(0).to(device)
            prob = bat(audio).item()
        
        # Clean up
        os.remove(temp_input)
        
        return {
            "has_watermark": prob >= 0.5,
            "confidence": prob
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
