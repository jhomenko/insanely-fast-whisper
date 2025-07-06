import torch
import numpy as np
from typing import List, Dict, Union, Optional

def get_speech_timestamps(audio: np.ndarray, sampling_rate: int, threshold: float = 0.5, 
                         min_speech_duration_ms: int = 250, max_speech_duration_s: float = 15.0, 
                         min_silence_duration_ms: int = 200, speech_pad_ms: int = 30) -> List[Dict[str, int]]:
    """
    Detect speech segments in audio using Silero VAD model.
    
    Args:
        audio: One dimensional float array of audio samples.
        sampling_rate: Sampling rate of the audio (expected to be 16000 Hz).
        threshold: Probability threshold for speech detection (0 to 1).
        min_speech_duration_ms: Minimum duration of speech segment in milliseconds.
        max_speech_duration_s: Maximum duration of speech segment in seconds.
        min_silence_duration_ms: Minimum duration of silence between speech segments in milliseconds.
        speech_pad_ms: Padding to add to speech segments in milliseconds.
    
    Returns:
        List of dictionaries with 'start' and 'end' sample indices of speech segments.
    """
    if sampling_rate != 16000:
        raise ValueError("Silero VAD expects audio with 16000 Hz sampling rate.")
    
    # Load Silero VAD model
    vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', 
                                         force_reload=False, trust_repo=True)
    get_speech_timestamps_func, _, _, _, _ = vad_utils
    
    # Convert audio to tensor
    audio_tensor = torch.from_numpy(audio).float()
    
    # Detect speech timestamps
    timestamps = get_speech_timestamps_func(
        audio_tensor,
        vad_model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    return timestamps

def apply_vad_filter(audio: np.ndarray, sampling_rate: int, speech_timestamps: List[Dict[str, int]]) -> np.ndarray:
    """
    Apply VAD filter to maintain original timing by injecting silence in non-speech segments.
    
    Args:
        audio: Original audio array.
        sampling_rate: Sampling rate of the audio.
        speech_timestamps: List of dictionaries with 'start' and 'end' sample indices of speech segments.
    
    Returns:
        Filtered audio array with silence in non-speech segments, maintaining original length.
    """
    if len(speech_timestamps) == 0:
        print("No speech detected, returning silence-filled audio of original length.")
        return np.zeros_like(audio)
    
    # Create output array of same length as input, initialized with silence
    filtered_audio = np.zeros_like(audio)
    
    # Copy speech segments to their original positions
    for segment in speech_timestamps:
        start_sample = segment['start']
        end_sample = segment['end']
        if start_sample < len(audio) and end_sample <= len(audio):
            filtered_audio[start_sample:end_sample] = audio[start_sample:end_sample]
    
    return filtered_audio
