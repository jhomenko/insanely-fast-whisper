import os
import torch
import numpy as np
from typing import Optional

def get_hdemucs_separator(device_id: str = "0") -> object:
    """
    Load the Hybrid Demucs model for vocal separation using torchaudio's pre-trained pipeline.
    
    Args:
        device_id: Device ID from CLI argument to determine the device for processing.
    
    Returns:
        Hybrid Demucs model instance for vocal separation.
    """
    try:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
        from torchaudio.transforms import Fade
        print("Debug: torchaudio library and HDEMUCS pipeline imported successfully.")
        
        # Device selection
        if device_id == "xpu" and torch.xpu.is_available():
            device = "xpu"
        elif device_id == "mps" and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = f"cuda:{device_id}" if device_id.isdigit() else "cuda:0"
        else:
            device = "cpu"
        print(f"Loading Hybrid Demucs model on {device}...")
        
        # Load the pre-trained model
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model = bundle.get_model()
        model.to(device)
        print(f"Hybrid Demucs model loaded successfully on {device}. Sample rate: {bundle.sample_rate} Hz")
        
        return {"model": model, "sample_rate": bundle.sample_rate, "device": device, "fade": Fade}
    except (ImportError, Exception) as e:
        print(f"Debug: Failed to import torchaudio or load HDEMUCS model: {str(e)}")
        raise ImportError("torchaudio library not detected or incompatible. Ensure 'torchaudio' is installed correctly (e.g., via 'pip install torchaudio'). Attempting fallback to original audio.")

def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None, fade_class=None):
    """
    Apply model to a given mixture. Use fade, and add segments together to process audio segment by segment.
    
    Args:
        model: The loaded Hybrid Demucs model.
        mix: Input audio tensor of shape (batch, channels, length).
        segment: Segment length in seconds.
        overlap: Overlap ratio between segments.
        device: Device to perform computation on.
        fade_class: Fade transform class for overlap handling.
    
    Returns:
        Separated sources tensor.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape
    sample_rate = 44100  # As per HDEMUCS_HIGH_MUSDB_PLUS sample rate
    
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = fade_class(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def apply_hdemucs_vocal_separation(audio: np.ndarray, sampling_rate: int, model: Optional[object] = None, device_id: str = "0") -> np.ndarray:
    """
    Apply vocal separation to audio using torchaudio's Hybrid Demucs pipeline.
    
    Args:
        audio: Input audio array.
        sampling_rate: Sampling rate of the audio.
        model: Pre-loaded Hybrid Demucs model instance, if available.
        device_id: Device ID from CLI argument to determine the device for processing.
    
    Returns:
        Processed audio array containing only vocals.
    """
    original_audio = audio.copy()
    
    if model is None:
        try:
            model_info = get_hdemucs_separator(device_id)
            model = model_info["model"]
            target_sample_rate = model_info["sample_rate"]
            device = model_info["device"]
            fade_class = model_info["fade"]
        except Exception as e:
            print(f"Unable to load Hybrid Demucs model due to: {str(e)}. Proceeding with original audio as fallback.")
            print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
            return original_audio
    else:
        target_sample_rate = 44100  # Default for HDEMUCS_HIGH_MUSDB_PLUS
        device = model.device
        fade_class = torch.transforms.Fade if not hasattr(model, "fade") else model.fade
    
    print("Separating vocals from background music using Hybrid Demucs...")
    print(f"Input audio stats - Length: {len(audio)} samples, Shape: {audio.shape}, Sample rate: {sampling_rate} Hz")
    
    if len(audio) == 0:
        print("Error: Input audio is empty. Cannot process empty audio. Proceeding with original audio as fallback.")
        return original_audio
    
    # Resample if necessary to match model's expected sample rate (44100 Hz)
    if sampling_rate != target_sample_rate:
        try:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float()
            if len(audio.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension for mono
            elif audio.shape[-1] == 2:
                audio_tensor = torch.transpose(audio_tensor, 0, 1)  # Convert to (channels, samples)
            audio_tensor = torchaudio.transforms.Resample(sampling_rate, target_sample_rate)(audio_tensor)
            audio = audio_tensor.numpy()
            if len(audio.shape) == 2:
                audio = audio.T  # Convert back to (samples, channels) if needed
            sampling_rate = target_sample_rate
            print(f"Resampled audio to {sampling_rate} Hz to match Hybrid Demucs requirements. New shape: {audio.shape}")
        except Exception as e:
            print(f"Error resampling audio: {str(e)}. Proceeding with original audio as fallback.")
            return original_audio
    
    # Ensure audio is in stereo (2 channels) format as expected by Hybrid Demucs
    if len(audio.shape) == 1:
        print("Converting mono audio to stereo by duplicating channel as required by Hybrid Demucs.")
        audio = np.stack([audio, audio], axis=-1)
        print(f"Converted audio shape to {audio.shape}")
    elif len(audio.shape) > 2:
        print(f"Warning: Input audio has unexpected shape {audio.shape}. Expected 1D (mono) or 2D (stereo). Attempting to flatten or convert.")
        if audio.shape[0] > 2:  # Likely (channels, samples)
            audio = audio.T  # Transpose to (samples, channels)
        audio = audio[:, :2] if audio.shape[-1] > 2 else audio  # Take first 2 channels if more are present
        print(f"Converted audio shape to {audio.shape}")
    elif audio.shape[-1] == 1:
        print("Converting mono audio to stereo by duplicating channel as required by Hybrid Demucs.")
        audio = np.stack([audio[:, 0], audio[:, 0]], axis=-1)
        print(f"Converted audio shape to {audio.shape}")
    
    try:
        # Convert numpy array to torch tensor with shape (batch, channels, length)
        audio_tensor = torch.from_numpy(audio).float().to(device)
        if audio_tensor.shape[-1] == 2:  # If (samples, channels)
            audio_tensor = audio_tensor.T.unsqueeze(0)  # Convert to (1, channels, samples)
        else:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, samples) for mono (shouldn't happen due to stereo conversion)
        print(f"Audio tensor shape for processing: {audio_tensor.shape}")
        
        # Process audio using Hybrid Demucs with chunking to manage memory
        sources = separate_sources(model, audio_tensor, segment=10.0, overlap=0.1, device=device, fade_class=fade_class)
        
        # Extract vocals (index 2 in model.sources for HDEMUCS_HIGH_MUSDB_PLUS, typically ['drums', 'bass', 'vocals', 'other'])
        sources_list = model.sources if hasattr(model, 'sources') else ['drums', 'bass', 'vocals', 'other']
        vocals_idx = sources_list.index('vocals') if 'vocals' in sources_list else 2
        vocals_tensor = sources[0, vocals_idx]  # Shape (channels, samples)
        print(f"Vocal separation successful. Vocals tensor shape: {vocals_tensor.shape}")
        
        # Convert to mono for Whisper compatibility
        if vocals_tensor.shape[0] == 2:
            vocals_tensor = torch.mean(vocals_tensor, dim=0, keepdim=False)  # Average channels to mono
            print(f"Converted vocals to mono, final shape: {vocals_tensor.shape}")
        
        # Resample back to 16000 Hz for Whisper compatibility
        target_sample_rate_whisper = 16000
        if target_sample_rate != target_sample_rate_whisper:
            try:
                import torchaudio
                # Ensure resampling is done on the same device as the tensor
                device = vocals_tensor.device
                resampler = torchaudio.transforms.Resample(target_sample_rate, target_sample_rate_whisper).to(device)
                vocals_tensor = resampler(vocals_tensor.unsqueeze(0)).squeeze(0)
                print(f"Resampled vocals back to {target_sample_rate_whisper} Hz for Whisper compatibility on device {device}, final shape: {vocals_tensor.shape}")
            except Exception as e:
                print(f"Error resampling vocals back to {target_sample_rate_whisper} Hz: {str(e)}. Proceeding without resampling, but this may cause issues with Whisper.")
        
        # Convert back to numpy array
        vocals = vocals_tensor.cpu().numpy()
        print(f"Vocal separation completed. Final vocals shape: {vocals.shape}, Sample rate: {target_sample_rate_whisper} Hz")
        
        return vocals
    except Exception as e:
        print(f"Error processing audio with Hybrid Demucs model: {str(e)}. Proceeding with original audio as fallback.")
        print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
        return original_audio
