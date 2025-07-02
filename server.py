import os
import argparse
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
import librosa
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor
import intel_extension_for_pytorch as ipex
from dotenv import load_dotenv

# Load environment variables for configuration
load_dotenv()

app = FastAPI(title="Insanely Fast Whisper API", description="API for audio transcription using optimized Whisper model")

# Global variables to store model and processor
model = None
processor = None
device = None

@app.on_event("startup")
async def startup_event():
    """Load the optimized Whisper model on startup."""
    global model, processor, device
    
    # Parse command line arguments for device override
    parser = argparse.ArgumentParser(description="FastAPI server for Whisper transcription")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to run the server on")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to run the server on")
    parser.add_argument("--xpu", action="store_true", help="Force use of XPU device")
    args, _ = parser.parse_known_args()
    
    # Determine device
    if args.xpu:
        device = "xpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model_name = os.getenv("MODEL_NAME", "openai/whisper-large-v3")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_cache=True
    )
    model.eval()
    model = ipex.llm.optimize(model, dtype=torch.float16, device=device)
    model.to(device)
    print(f"Model loaded and optimized on {device}")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(model_name)
    print("Processor loaded")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """Endpoint to transcribe audio files."""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again later.")
    
    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".m4a"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Only {', '.join(allowed_extensions)} are allowed.")
    
    # Validate file size (limit to 20 MB)
    file_size = file.size
    if file_size > 20 * 1024 * 1024:  # 20 MB in bytes
        raise HTTPException(status_code=400, detail="File size exceeds 20 MB limit.")
    
    # Save file to temporary directory
    temp_dir = "/tmp/audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and process audio
        audio, sr = librosa.load(temp_file_path, sr=16000, mono=True)
        audio_duration = len(audio) / sr
        
        # Validate audio duration (limit to 1 minute)
        if audio_duration > 60:
            raise HTTPException(status_code=400, detail="Audio duration exceeds 1 minute limit.")
        
        # Process audio for transcription
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device, dtype=torch.float16)
        
        # Generate transcription
        start_time = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        processing_time = time.time() - start_time
        
        # Return response
        return JSONResponse(content={
            "text": transcription,
            "duration_sec": audio_duration,
            "chunks": 1,
            "timestamped": False
        })
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI server for Whisper transcription")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to run the server on")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to run the server on")
    parser.add_argument("--xpu", action="store_true", help="Force use of XPU device")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
