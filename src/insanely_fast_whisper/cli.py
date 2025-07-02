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
        use_cache=True,
        use_auth_token=args.hf_token if args.hf_token != "no_token" else None
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
        processor = WhisperProcessor.from_pretrained(
            args.model_name,
            use_auth_token=args.hf_token if args.hf_token != "no_token" else None
        )

        import librosa
        import os
        # Check if file_name is a relative path and prepend 'input' folder if so
        if not os.path.isabs(args.file_name) and not args.file_name.startswith('http'):
            args.file_name = os.path.join('input', args.file_name)
        audio, sr = librosa.load(args.file_name, sr=16000)
        audio_duration = len(audio) / sr
        print(f"Audio loaded from {args.file_name}, length: {audio_duration:.2f} seconds")

        # Process audio input for long-form transcription
        start_time = time.time()
        print("Starting transcription process...")
        model.generation_config.max_new_tokens = 256
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True
        )
        input_features = inputs.input_features.to(device, dtype=torch.float16)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
        print(f"Input features moved to device: {device} with dtype: {input_features.dtype}")

        # Update generate_kwargs with long-form transcription parameters
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

        # Generate transcription using the model's generate method directly
        with torch.no_grad():
            if attention_mask is not None:
                predicted_ids = model.generate(input_features, attention_mask=attention_mask, **generate_kwargs)
            else:
                predicted_ids = model.generate(input_features, **generate_kwargs)
        print(f"Transcription completed in {time.time() - start_time:.2f} seconds")

        # Decode the transcription with timestamp information
        result = processor.batch_decode(predicted_ids, skip_special_tokens=False)
        clean_result = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        # Process transcription output to build chunks using timestamps
        chunks = []
        if isinstance(result, list) and len(result) == 1:
            # Single audio input, attempt to parse timestamps from the raw output
            raw_output = result[0]
            import re
            
            # Look for timestamp patterns in the raw output (e.g., <|0.00|>, <|1.23|>)
            timestamp_pattern = r"<\|(\d+\.\d+)\|>"
            timestamps = re.findall(timestamp_pattern, raw_output)
            text_parts = re.split(timestamp_pattern, raw_output)
            
            if len(timestamps) > 0:
                # We have timestamps, build chunks accordingly
                used_texts = set()  # To avoid repetitions
                for i in range(len(timestamps)):
                    start_time = float(timestamps[i])
                    # Get the text between this timestamp and the next (or end)
                    text_segment = text_parts[i + 1].strip()
                    # Clean up any remaining special tokens or artifacts
                    text_segment = re.sub(r"<\|[^>]*\|>", "", text_segment).strip()
                    if text_segment and text_segment not in used_texts:
                        used_texts.add(text_segment)
                        end_time = float(timestamps[i + 1]) if i + 1 < len(timestamps) else audio_duration
                        chunks.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text_segment
                        })
            else:
                # No timestamps found, use the clean transcription as a single chunk
                clean_text = clean_result[0] if isinstance(clean_result, list) else clean_result
                chunks.append({
                    'start': 0.0,
                    'end': audio_duration,
                    'text': clean_text
                })
        else:
            # Fallback if output format is unexpected
            clean_text = clean_result[0] if isinstance(clean_result, list) else clean_result
            chunks.append({
                'start': 0.0,
                'end': audio_duration,
                'text': clean_text
            })
        
        # Build the text field by concatenating chunk texts
        text = " ".join(chunk['text'] for chunk in chunks)
        outputs = {'chunks': chunks, 'segments': chunks, 'text': text}
        print(f"Transcription completed with {len(chunks)} chunks")

    if args.hf_token != "no_token":
        speakers_transcript = diarize(args, outputs)
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
