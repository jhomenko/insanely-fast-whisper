import requests
import torch
import numpy as np
from torchaudio import functional as F
from transformers.pipelines.audio_utils import ffmpeg_read
import sys


# Code lifted from https://github.com/huggingface/speechbox/blob/main/src/speechbox/diarize.py
# and from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py


def preprocess_inputs(inputs):
    import os
    import subprocess
    import tempfile

    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            # Check if the file extension is .m4a or other potentially unsupported formats
            if os.path.splitext(inputs)[1].lower() in ['.m4a']:
                # Create a temporary .wav file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # Convert .m4a to .wav using ffmpeg
                    subprocess.run(['ffmpeg', '-i', inputs, '-ac', '1', '-ar', '16000', '-y', temp_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with open(temp_path, 'rb') as f:
                        inputs = f.read()
                    # Clean up temporary file
                    os.unlink(temp_path)
            else:
                with open(inputs, 'rb') as f:
                    inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs.copy()).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def diarize_audio(diarizer_inputs, diarization_pipeline, num_speakers, min_speakers, max_speakers):
    diarization = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": 16000},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )
    print(new_segments)
    return new_segments


def post_process_segments_and_transcripts(new_segments, transcript, group_by_speaker) -> list:
    # Handle the transcript structure based on pipeline output
    if isinstance(transcript, dict) and "chunks" in transcript:
        chunks = transcript["chunks"]
    elif isinstance(transcript, list):
        chunks = transcript
    else:
        return []

    # get the end timestamps for each chunk from the ASR output
    import sys
    end_timestamps = np.array(
        [chunk["timestamp"][1] if "timestamp" in chunk and len(chunk["timestamp"]) > 1 and chunk["timestamp"][1] is not None else sys.float_info.max for chunk in chunks]
    )
    segmented_preds = []
    
    # Keep track of original chunks and timestamps for handling remaining chunks
    original_chunks = chunks.copy()

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        if len(end_timestamps) == 0:
            break
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in chunks[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        chunks[0]["timestamp"][0] if "timestamp" in chunks[0] and len(chunks[0]["timestamp"]) > 0 else 0.0,
                        chunks[upto_idx]["timestamp"][1] if "timestamp" in chunks[upto_idx] and len(chunks[upto_idx]["timestamp"]) > 1 else 0.0,
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **chunks[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        chunks = chunks[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

    # Handle remaining chunks that weren't assigned to any speaker segment
    # This fixes the issue where the last chunk gets cut off
    if len(chunks) > 0 and len(new_segments) > 0:
        # Assign remaining chunks to the last speaker
        last_speaker = new_segments[-1]["speaker"]
        print(f"Found {len(chunks)} remaining chunks after diarization. Assigning to last speaker: {last_speaker}")
        
        if group_by_speaker:
            # If we're grouping by speaker, we need to extend the last segment's text and timestamp
            if segmented_preds:
                # Extend the last segment
                remaining_text = "".join([chunk["text"] for chunk in chunks])
                segmented_preds[-1]["text"] += remaining_text
                
                # Update the end timestamp to include the last chunk
                if chunks and "timestamp" in chunks[-1] and len(chunks[-1]["timestamp"]) > 1:
                    segmented_preds[-1]["timestamp"] = (
                        segmented_preds[-1]["timestamp"][0],
                        chunks[-1]["timestamp"][1]
                    )
        else:
            # If not grouping by speaker, add each remaining chunk individually
            for chunk in chunks:
                segmented_preds.append({"speaker": last_speaker, **chunk})

    return segmented_preds
