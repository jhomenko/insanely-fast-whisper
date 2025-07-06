import torch
from pyannote.audio import Pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from .diarize import post_process_segments_and_transcripts, diarize_audio, \
    preprocess_inputs


def diarize(args, outputs, audio_duration=None):
    # Workaround for PyTorch unpickling error with weights_only=True in newer versions
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    # Add pyannote.audio.core.task.Resolution to safe globals as required by error message
    try:
        from pyannote.audio.core.task import Resolution, Specifications, Problem
        torch.serialization.add_safe_globals([Resolution, Specifications, Problem])
    except ImportError:
        print("Could not import Resolution, Specifications, or Problem from pyannote.audio.core.task for safe globals.")
    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path=args.diarization_model,
        use_auth_token=args.hf_token,
    )
    if args.device_id == "mps":
        device = "mps"
    elif args.device_id == "xpu":
        device = "xpu"
    else:
        device = f"cuda:{args.device_id}"
    diarization_pipeline.to(torch.device(device))

    with Progress(
            TextColumn("🤗 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Segmenting...", total=None)

        inputs, diarizer_inputs = preprocess_inputs(inputs=args.file_name)

        segments = diarize_audio(diarizer_inputs, diarization_pipeline, args.num_speakers, args.min_speakers, args.max_speakers)

    # Limit diarization segments to match the audio duration used for transcription if provided
    if audio_duration is not None:
        new_segments = [
            seg for seg in new_segments
            if seg["segment"]["start"] <= audio_duration
        ]
        for seg in new_segments:
            if seg["segment"]["end"] > audio_duration:
                seg["segment"]["end"] = audio_duration
        print(f"Limited diarization segments to match audio duration of {audio_duration:.2f} seconds. Resulting segments: {len(new_segments)}")

    return post_process_segments_and_transcripts(
        new_segments, outputs["chunks"], group_by_speaker=False
    )
