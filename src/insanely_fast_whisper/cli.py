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
    default=0.5,
    help="Threshold for Voice Activity Detection (between 0 and 1). Higher values are stricter. (default: 0.5)",
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

    ts = True if args.timestamp == "chunk" else "word"

    language = None if args.language == "None" else args.language

    generate_kwargs = {"task": args.task, "language": language}

    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)
        print("Progress bar initialized.")

        from transformers import WhisperProcessor, pipeline
        print("Loading WhisperProcessor...")
        processor = WhisperProcessor.from_pretrained(
            args.model_name,
            use_auth_token=args.hf_token if args.hf_token != "no_token" else None
        )
        print("WhisperProcessor loaded.")

        import librosa
        import os
        # Check if file_name is a relative path and prepend 'input' folder if so
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

        # Voice Activity Detection (VAD) pre-processing if enabled
        speech_chunks = []
        if args.vad_filter:
            print("Applying Voice Activity Detection to filter silent parts...")
            try:
                vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, trust_repo=True)
                get_speech_timestamps, _, _, _, _ = vad_utils
                audio_tensor = torch.from_numpy(audio).float()
                speech_timestamps = get_speech_timestamps(
                    audio_tensor,
                    vad_model,
                    sampling_rate=sr,
                    threshold=args.vad_threshold,
                    min_speech_duration_ms=args.vad_min_speech_duration_ms,
                    max_speech_duration_s=args.vad_max_speech_duration_s,
                    min_silence_duration_ms=args.vad_min_silence_duration_ms,
                    speech_pad_ms=args.vad_speech_pad_ms
                )
                speech_chunks = [(ts['start'] / sr, ts['end'] / sr) for ts in speech_timestamps]
                if not speech_chunks:
                    print("No speech detected after VAD. Proceeding with full audio.")
                    speech_chunks = [(0.0, audio_duration)]
                else:
                    print(f"Detected {len(speech_chunks)} speech segments after VAD.")
            except Exception as e:
                print(f"Failed to apply VAD: {str(e)}. Proceeding with full audio.")
                speech_chunks = [(0.0, audio_duration)]
        else:
            speech_chunks = [(0.0, audio_duration)]

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

        # Update generate_kwargs for long-form transcription
        generate_kwargs.update({
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 1.35,
            "return_timestamps": True if args.timestamp == "chunk" else "word",
            "max_new_tokens": 256,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "no_speech_threshold": 0.6
        })

        # Process audio using the pipeline with built-in chunking
        print("Processing audio with pipeline...")
        try:
            result = pipe(audio, generate_kwargs=generate_kwargs)
            print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return

        # Extract segments from the pipeline result
        if "chunks" in result:
            outputs = {
                'segments': [
                    {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
                    for chunk in result["chunks"]
                    if chunk["timestamp"][0] is not None and chunk["timestamp"][1] is not None
                ],
                'text': result.get("text", "")
            }
        else:
            outputs = {
                'segments': [{"start": 0.0, "end": audio_duration, "text": result["text"]}],
                'text': result["text"]
            }
        # Save transcription output to a separate JSON file for review
        with open("output/transcript.json", "w", encoding="utf8") as fp:
            json.dump(outputs, fp, ensure_ascii=False)
        print("Transcription output saved to output/transcript.json for schema review.")

    if args.hf_token != "no_token":
        speakers_transcript = diarize(args, outputs)
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
