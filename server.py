import os
import argparse
import tempfile
import shutil
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Import the CLI module directly to reuse its functionality
from insanely_fast_whisper.cli import main as cli_main, parser as cli_parser

# Load environment variables for configuration
load_dotenv()

app = FastAPI(title="Insanely Fast Whisper API", description="API for audio transcription using optimized Whisper model")

# Global variable to store command-line arguments
args = None

@app.on_event("startup")
async def startup_event():
    """Initialize server settings on startup."""
    global args
    
    # Parse command line arguments for server and CLI settings
    parser = argparse.ArgumentParser(description="FastAPI server for Whisper transcription")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to run the server on")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to run the server on")
    
    # Add all arguments from cli.py parser to ensure compatibility
    for action in cli_parser._actions:
        if action.dest not in ['help', 'file_name', 'transcript_path']:  # Exclude file-specific args to be set per request
            parser.add_argument(
                action.option_strings[0] if action.option_strings else action.dest,
                dest=action.dest,
                type=action.type,
                default=action.default if action.default is not None else os.getenv(action.dest.upper(), action.default),
                help=action.help,
                choices=action.choices
            )
    
    args, _ = parser.parse_known_args()
    print(f"Server initialized with host: {args.host}, port: {args.port}")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """Endpoint to transcribe audio files using the CLI module directly."""
    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".m4a"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Only {', '.join(allowed_extensions)} are allowed.")
    
    # Save file to temporary directory
    temp_dir = "/tmp/audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    temp_output_path = os.path.join(temp_dir, f"output_{file.filename}.json")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare command-line arguments for CLI main function by overriding sys.argv
        original_argv = sys.argv
        sys.argv = [
            "cli.py",  # Script name placeholder
            "--file-name", temp_file_path,
            "--transcript-path", temp_output_path
        ]
        
        # Add all relevant arguments from the server configuration to mirror CLI invocation
        if hasattr(args, 'device_id') and args.device_id:
            sys.argv.extend(["--device-id", str(args.device_id)])
        if hasattr(args, 'model_name') and args.model_name:
            sys.argv.extend(["--model-name", args.model_name])
        if hasattr(args, 'hf_token') and args.hf_token and args.hf_token != "no_token":
            sys.argv.extend(["--hf-token", args.hf_token])
        if hasattr(args, 'task') and args.task:
            sys.argv.extend(["--task", args.task])
        if hasattr(args, 'language') and args.language and args.language != "None":
            sys.argv.extend(["--language", args.language])
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            sys.argv.extend(["--batch-size", str(args.batch_size)])
        if hasattr(args, 'flash') and args.flash is not None:
            sys.argv.extend(["--flash", str(args.flash).lower()])
        if hasattr(args, 'vad_filter') and args.vad_filter is not None:
            sys.argv.extend(["--vad-filter", str(args.vad_filter).lower()])
        if hasattr(args, 'vad_threshold') and args.vad_threshold is not None:
            sys.argv.extend(["--vad-threshold", str(args.vad_threshold)])
        if hasattr(args, 'vad_min_speech_duration_ms') and args.vad_min_speech_duration_ms is not None:
            sys.argv.extend(["--vad-min-speech-duration-ms", str(args.vad_min_speech_duration_ms)])
        if hasattr(args, 'vad_max_speech_duration_s') and args.vad_max_speech_duration_s is not None:
            sys.argv.extend(["--vad-max-speech-duration-s", str(args.vad_max_speech_duration_s)])
        if hasattr(args, 'vad_min_silence_duration_ms') and args.vad_min_silence_duration_ms is not None:
            sys.argv.extend(["--vad-min-silence-duration-ms", str(args.vad_min_silence_duration_ms)])
        if hasattr(args, 'vad_speech_pad_ms') and args.vad_speech_pad_ms is not None:
            sys.argv.extend(["--vad-speech-pad-ms", str(args.vad_speech_pad_ms)])
        if hasattr(args, 'timestamp') and args.timestamp:
            sys.argv.extend(["--timestamp", args.timestamp])
        if hasattr(args, 'diarization_model') and args.diarization_model:
            sys.argv.extend(["--diarization_model", args.diarization_model])
        if hasattr(args, 'num_speakers') and args.num_speakers is not None:
            sys.argv.extend(["--num-speakers", str(args.num_speakers)])
        if hasattr(args, 'min_speakers') and args.min_speakers is not None:
            sys.argv.extend(["--min-speakers", str(args.min_speakers)])
        if hasattr(args, 'max_speakers') and args.max_speakers is not None:
            sys.argv.extend(["--max-speakers", str(args.max_speakers)])
        
        # Redirect stdout and stderr to capture output and avoid display conflicts
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                cli_main()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
        
        # Read the output JSON file generated by cli.py and return its contents directly
        if os.path.exists(temp_output_path):
            with open(temp_output_path, 'r', encoding='utf8') as f:
                result = json.load(f)
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail="Transcription output file not found.")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI server for Whisper transcription")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to run the server on")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to run the server on")
    
    # Add all arguments from cli.py parser to ensure compatibility
    for action in cli_parser._actions:
        if action.dest not in ['help', 'file_name', 'transcript_path']:  # Exclude file-specific args to be set per request
            parser.add_argument(
                action.option_strings[0] if action.option_strings else action.dest,
                dest=action.dest,
                type=action.type,
                default=action.default if action.default is not None else os.getenv(action.dest.upper(), action.default),
                help=action.help,
                choices=action.choices
            )
    
    args = parser.parse_args()
    
    # Validate speaker arguments
    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("--num-speakers cannot be used together with --min-speakers or --max-speakers.")
    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1.")
    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be at least 1.")
    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be at least 1.")
    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        parser.error("--min-speakers cannot be greater than --max-speakers.")
    
    uvicorn.run(app, host=args.host, port=args.port)
