import json
import argparse
import time
import torch
from transformers import WhisperProcessor
from transformers import AutoModelForSpeechSeq2Seq
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import librosa
import numpy as np

from insanely_fast_whisper.utils.diarization_pipeline import diarize
from insanely_fast_whisper.utils.result import build_result

parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=True,
    type=str,
    help="Path or URL to the audio file to be transcribed. If a relative path is provided, it will be looked for in the 'input' folder.",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device ID for your GPU. Just pass the device number when using CUDA, "mps" for Macs with Apple Silicon, or "xpu" for Intel XPU devices. (default: "0")',
)
parser.add_argument(
    "--transcript-path",
    required=False,
    default="output/output.json",
    type=str,
    help="Path to save the transcription output. (default: output/output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="distil-whisper/distil-medium.en",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: distil-whisper/distil-medium.en)",
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=12,
    help="Number of parallel batches for chunk processing. Reduce if you face OOMs. (default: 12)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
parser.add_argument(
    "--vad-filter",
    required=False,
    type=bool,
    default=False,
    help="Enable Voice Activity Detection to filter out silent parts before transcription. (default: False)",
)
parser.add_argument(
    "--vad-threshold",
    required=False,
    type=float,
    default=0.3,
    help="Threshold for Voice Activity Detection (between 0 and 1). Lower values are more lenient. (default: 0.3)",
)
parser.add_argument(
    "--vad-min-speech-duration-ms",
    required=False,
    type=int,
    default=250,
    help="Minimum speech duration in milliseconds for VAD. (default: 250)",
)
parser.add_argument(
    "--vad-max-speech-duration-s",
    required=False,
    type=float,
    default=15.0,
    help="Maximum speech duration in seconds for VAD. (default: 15.0)",
)
parser.add_argument(
    "--vad-min-silence-duration-ms",
    required=False,
    type=int,
    default=200,
    help="Minimum silence duration in milliseconds between speech segments for VAD. (default: 200)",
)
parser.add_argument(
    "--vad-speech-pad-ms",
    required=False,
    type=int,
    default=30,
    help="Padding added to speech segments in milliseconds for VAD. (default: 30)",
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)
parser.add_argument(
    "--hf-token",
    required=False,
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
)
parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)
parser.add_argument(
    "--num-speakers",
    required=False,
    default=None,
    type=int,
    help="Specifies the exact number of speakers present in the audio file. Useful when the exact number of participants in the conversation is known. Must be at least 1. Cannot be used together with --min-speakers or --max-speakers. (default: None)",
)
parser.add_argument(
    "--min-speakers",
    required=False,
    default=None,
    type=int,
    help="Sets the minimum number of speakers that the system should consider during diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be less than or equal to --max-speakers if both are specified. (default: None)",
)
parser.add_argument(
    "--max-speakers",
    required=False,
    default=None,
    type=int,
    help="Defines the maximum number of speakers that the system should consider in diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be greater than or equal to --min-speakers if both are specified. (default: None)",
)
parser.add_argument(
    "--beam-size",
    required=False,
    default=1,
    type=int,
    help="Number of beams for beam search in transcription. Higher values may improve accuracy but increase computation time. (default: 1)",
)
parser.add_argument(
    "--temperature",
    required=False,
    default=0.0,
    type=float,
    help="Temperature for transcription output randomness. Lower values are more deterministic, higher values introduce diversity. (default: 0.0)",
)
parser.add_argument(
    "--vocal-removal",
    required=False,
    type=bool,
    default=False,
    help="Enable vocal removal to separate vocals from background music before transcription. (default: False)",
)
parser.add_argument(
    "--vocal-method",
    required=False,
    default="uvr",
    type=str,
    choices=["uvr", "hdemucs"],
    help="Method for vocal removal: 'uvr' for Ultimate Vocal Remover API or 'hdemucs' for torchaudio Hybrid Demucs. (default: uvr)",
)
parser.add_argument(
    "--vocal-model",
    required=False,
    type=str,
    default="UVR-MDX-NET-Inst_HQ_4",
    help="Name of the UVR model for vocal removal when using 'uvr' method. (default: UVR-MDX-NET-Inst_HQ_4)",
)
parser.add_argument(
    "--vocal-model-dir",
    required=False,
    type=str,
    default="./uvr_models",
    help="Directory to store UVR models for vocal removal when using 'uvr' method. (default: ./uvr_models)",
)

def main():
    args = parser.parse_args()
    print("Parsed command line arguments:", vars(args))

    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("--num-speakers cannot be used together with --min-speakers or --max-speakers.")

    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1.")

    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be at least 1.")

    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be at least 1.")

    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        if args.min_speakers > args.max_speakers:
            parser.error("--min-speakers cannot be greater than --max-speakers.")

    # Load the model using Transformers
    from transformers import AutoModelForSpeechSeq2Seq as TransformersAutoModel
    print("Loading model...")
    model = TransformersAutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        use_cache=True,
        use_auth_token=args.hf_token if args.hf_token != "no_token" else None
    )
    print("Model loaded successfully.")
    
    # Set model to eval mode before optimization
    model.eval()
    
    # Optimize the model with intel_extension_for_pytorch LLM module
    import intel_extension_for_pytorch as ipex
    if args.device_id == "mps":
        device = "mps"
    elif args.device_id == "xpu":
        device = "xpu"
    else:
        device = f"cuda:{args.device_id}"
    print("Optimizing model for device:", device)
    model = ipex.llm.optimize(model, dtype=torch.float16, device=device)
    model.to(device)
    print(f"Model moved to device: {device}")

    if args.flash:
        model.config.attn_implementation = "flash_attention_2"
    else:
        model.config.attn_implementation = "sdpa"

    from transformers import pipeline, WhisperProcessor
    
    # Load processor for tokenizer and feature extractor
    print("Loading WhisperProcessor...")
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        use_auth_token=args.hf_token if args.hf_token != "no_token" else None
    )
    print("WhisperProcessor loaded.")
    
    # Set up the pipeline for long-form transcription with built-in chunking
    start_time = time.time()
    print("Starting transcription process using built-in chunking...")
    print("Setting up pipeline...")
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=25,  # Set chunk length to 25 seconds
            stride_length_s=(5, 5),  # Set stride to 5 seconds for overlap (left, right)
            batch_size=args.batch_size,
            torch_dtype=torch.float16,
            device=device,
        )
        print("Pipeline setup completed.")
    except Exception as e:
        print(f"Error setting up pipeline: {str(e)}")
        return

    ts = True  # Use chunk-level timestamps, passed within generate_kwargs

    language = None if args.language == "None" else args.language

    generate_kwargs = {"task": args.task, "language": language}

    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    # Update generate_kwargs with beam size and temperature
    generate_kwargs.update({
        "num_beams": args.beam_size,
        "temperature": args.temperature
    })

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        # Check if file_name is a relative path and prepend 'input' folder if so
        import os
        import librosa
        if not os.path.isabs(args.file_name) and not args.file_name.startswith('http'):
            args.file_name = os.path.join('input', args.file_name)
        
        print(f"Loading audio file: {args.file_name}")
        try:
            audio, sr = librosa.load(args.file_name, sr=16000)
            audio_duration = len(audio) / sr
            print(f"Audio loaded from {args.file_name}, length: {audio_duration:.2f} seconds")
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            return

        # Pre-process audio with VAD if enabled
        if args.vad_filter:
            try:
                from insanely_fast_whisper.utils.silero import get_speech_timestamps, apply_vad_filter
                print("Applying Voice Activity Detection to filter silent parts...")
                speech_timestamps = get_speech_timestamps(
                    audio,
                    sr,
                    threshold=args.vad_threshold,
                    min_speech_duration_ms=args.vad_min_speech_duration_ms,
                    max_speech_duration_s=args.vad_max_speech_duration_s,
                    min_silence_duration_ms=args.vad_min_silence_duration_ms,
                    speech_pad_ms=args.vad_speech_pad_ms
                )
                if not speech_timestamps:
                    print("No speech detected after VAD. Proceeding with full audio.")
                else:
                    print(f"Detected {len(speech_timestamps)} speech segments after VAD.")
                    audio = apply_vad_filter(audio, sr, speech_timestamps)
            except Exception as e:
                print(f"Failed to apply VAD: {str(e)}. Proceeding with original audio.")

        # Pre-process audio with vocal removal if enabled
        if args.vocal_removal:
            try:
                if args.vocal_method == "uvr":
                    from insanely_fast_whisper.utils.uvr import apply_vocal_removal
                    print("Applying vocal removal using UVR method to isolate speech...")
                    audio = apply_vocal_removal(audio, sr, model_name=args.vocal_model, model_dir=args.vocal_model_dir, device_id=args.device_id)
                else:  # hdemucs
                    try:
                        from insanely_fast_whisper.utils.hdemucs_direct import apply_hdemucs_vocal_separation
                        print("Applying vocal removal using direct Hybrid Transformer Demucs method to isolate speech...")
                        audio = apply_hdemucs_vocal_separation(audio, sr, device_id=args.device_id, model_name="htdemucs_ft")
                    except ImportError as ie:
                        print(f"Direct Demucs implementation not available: {str(ie)}. Falling back to torchaudio implementation.")
                        from insanely_fast_whisper.utils.hdemucs import apply_hdemucs_vocal_separation
                        print("Applying vocal removal using Hybrid Demucs method to isolate speech...")
                        audio = apply_hdemucs_vocal_separation(audio, sr, device_id=args.device_id)
                print("Vocal removal completed.")
                # Update audio duration after vocal removal for consistency in subsequent steps
                audio_duration = len(audio) / sr
                print(f"Updated audio duration after vocal removal: {audio_duration:.2f} seconds")
            except Exception as e:
                print(f"Failed to apply vocal removal: {str(e)}. Proceeding with original audio.")

        # Process audio using the pipeline with built-in chunking
        print("Processing audio with pipeline...")
        print(f"Audio input stats before transcription - Length: {len(audio)} samples, Sample rate: {sr} Hz")
        try:
            outputs = pipe(audio, generate_kwargs=generate_kwargs, return_timestamps=ts)
            print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
            print("Raw transcription output structure:", type(outputs))
            print("Raw transcription output content:", outputs)
            # Check if outputs is empty or not in expected format
            if not outputs:
                print("Error: Transcription output is empty. Saving empty result.")
                outputs = {"text": "", "chunks": []}
            elif not isinstance(outputs, dict):
                print("Error: Transcription output is not a dictionary. Saving empty result.")
                outputs = {"text": "", "chunks": []}
            elif "text" not in outputs:
                print("Error: 'text' key not found in transcription output. Saving empty result.")
                outputs = {"text": "", "chunks": []}
            elif "chunks" not in outputs or not outputs.get("chunks"):
                print("Warning: 'chunks' key missing or empty in transcription output. Adding empty chunks.")
                outputs["chunks"] = []
        except Exception as e:
            print(f"Error during transcription: {str(e)}. This may be due to parameter configuration issues with the pipeline. Detailed traceback:")
            import traceback
            traceback.print_exc()
            outputs = {"text": "", "chunks": []}

        # Save transcription output to a separate JSON file for review
        with open("output/transcript.json", "w", encoding="utf8") as fp:
            json.dump(outputs, fp, ensure_ascii=False)
        print("Transcription output saved to output/transcript.json for schema review.")

    if args.hf_token != "no_token":
        # Limit diarization to match the audio duration used for transcription if audio was limited
        speakers_transcript = diarize(args, outputs, audio_duration=audio_duration)
        # Save diarization output to a separate JSON file for review
        with open("output/diarization.json", "w", encoding="utf8") as fp:
            json.dump(speakers_transcript, fp, ensure_ascii=False)
        print("Diarization output saved to output/diarization.json for schema review.")
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed & speaker segmented, go check it out over here 👉 {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result([], outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed, go check it out over here 👉 {args.transcript_path}"
        )
