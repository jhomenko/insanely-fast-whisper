import json
import argparse
import time
from transformers import WhisperProcessor
from transformers import AutoModelForSpeechSeq2Seq
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import torch
import librosa
import numpy as np

from .utils.diarization_pipeline import diarize
from .utils.result import build_result

parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=True,
    type=str,
    help="Path or URL to the audio file to be transcribed.",
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
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
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
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 12)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
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
    model = TransformersAutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        use_cache=True
    )
    
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
    model = ipex.llm.optimize(model, dtype=torch.float16, device=device)
    model.to(device)
    print(f"Model moved to device: {device}")

    if args.flash:
        model.config.attn_implementation = "flash_attention_2"
    else:
        model.config.attn_implementation = "sdpa"

    ts = "word" if args.timestamp == "word" else True

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

        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(args.model_name)

        import librosa
        audio, sr = librosa.load(args.file_name, sr=16000)
        input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device, dtype=torch.float16)
        print(f"Input features moved to device: {device} with dtype: {input_features.dtype}")

        # Use manual chunking with direct generate method for long-form transcription
        start_time = time.time()
        print("Starting transcription process...")
        audio, sr = librosa.load(args.file_name, sr=16000)
        audio_duration = len(audio) / sr
        print(f"Audio loaded, length: {audio_duration:.2f} seconds")
        
        # Define chunk length (30 seconds is the max receptive field for Whisper)
        chunk_length_s = 30
        chunk_samples = chunk_length_s * sr
        num_chunks = int(np.ceil(audio_duration / chunk_length_s))
        print(f"Processing audio in {num_chunks} chunks of {chunk_length_s} seconds each")
        
        segments = []
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, len(audio))
            chunk_audio = audio[start_sample:end_sample]
            chunk_duration = (end_sample - start_sample) / sr
            
            print(f"Processing chunk {i+1}/{num_chunks} ({chunk_duration:.2f} seconds)")
            chunk_start_time = time.time()
            
            # Extract features for the chunk
            input_features = processor(chunk_audio, sampling_rate=sr, return_tensors="pt").input_features.to(device, dtype=torch.float16)
            
            # Generate transcription for the chunk
            with torch.no_grad():
                predicted_ids = model.generate(input_features, **generate_kwargs)
            
            # Decode the transcription
            chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Adjust timestamps to match the overall audio timeline
            chunk_start_time_audio = start_sample / sr
            segments.append({
                'start': chunk_start_time_audio,
                'end': chunk_start_time_audio + chunk_duration,
                'text': chunk_text
            })
            
            print(f"Chunk {i+1}/{num_chunks} processed in {time.time() - chunk_start_time:.2f} seconds")
        
        outputs = {'segments': segments}
        print(f"Transcription completed in {time.time() - start_time:.2f} seconds for all {num_chunks} chunks")

    if args.hf_token != "no_token":
        speakers_transcript = diarize(args, outputs)
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result([], outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed go check it out over here 👉 {args.transcript_path}"
        )
