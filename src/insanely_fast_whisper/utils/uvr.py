import os
import torch
import numpy as np
from typing import Optional, Tuple, Union

def get_vocal_separator(model_name: str = "hdemucs_mmi", model_dir: str = "./uvr_models", device_id: str = "0") -> object:
    """
    Load a UVR model for vocal separation using the ultimatevocalremover_api structure.
    
    Args:
        model_name: Name of the UVR model to use (default: hdemucs_mmi as per example).
        model_dir: Directory to store downloaded models (not directly used in this API but kept for compatibility).
        device_id: Device ID from CLI argument to determine if XPU is used for fallback logic.
    
    Returns:
        UVR model instance for vocal separation.
    """
    try:
        import uvr
        from uvr import models
        print(f"Debug: UVR library imported successfully from 'uvr'")
        
        # Map common model names from cli.py to UVR library compatible names if needed
        model_name_mapping = {
            "UVR-MDX-NET-Inst_HQ_4": "hdemucs_mmi",  # Default mapping if cli passes this name
        }
        if model_name in model_name_mapping:
            mapped_model_name = model_name_mapping[model_name]
            print(f"Mapping model name '{model_name}' to '{mapped_model_name}' for UVR compatibility.")
            model_name = mapped_model_name
        
        # Attempt to download models if functionality is available
        try:
            from uvr.utils.get_models import download_all_models
            import json
            import pkg_resources
            # Attempt to dynamically locate models.json within the uvr package or fallback paths
            models_json_path = None
            possible_paths = [
                os.path.join(model_dir, "models.json"),
                "/content/ultimatevocalremover_api/src/models_dir/models.json",
            ]
            try:
                # Try to find models.json in the installed package resources
                models_json_path = pkg_resources.resource_filename('uvr', 'models_dir/models.json')
            except (ImportError, KeyError):
                pass
            
            if not models_json_path or not os.path.exists(models_json_path):
                for path in possible_paths:
                    if os.path.exists(path):
                        models_json_path = path
                        break
            
            if models_json_path and os.path.exists(models_json_path):
                with open(models_json_path, "r") as f:
                    models_json = json.load(f)
                print(f"Downloading UVR models using configuration from {models_json_path}...")
                download_all_models(models_json)
            else:
                print(f"Warning: models.json not found in expected paths. Models will not be downloaded automatically. Checked paths: {possible_paths}")
                print("Please ensure models are downloaded manually from 'https://github.com/NextAudioGen/ultimatevocalremover_api' or associated sources.")
        except (ImportError, Exception) as e:
            print(f"Warning: Could not download models automatically due to: {str(e)}. Ensure models are available manually from 'https://github.com/NextAudioGen/ultimatevocalremover_api'.")
        
        # Device selection with fallback logic for XPU compatibility issues
        if device_id == "xpu":
            device = "cpu"  # Default to CPU for UVR processing when XPU is specified to avoid potential XPU issues
            print(f"Defaulting to CPU for UVR processing even though XPU is specified, to avoid potential compatibility or resource issues.")
        else:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading UVR model on {device}...")
        
        model_config = {
            "segment": 1,  # Reduced segment size to minimize memory usage
            "split": False  # Disable split to avoid excessive memory allocation
        }
        print(f"Model configuration: {model_config}")
        
        # Add safe globals for PyTorch weights loading to handle weights_only=True default in newer versions
        try:
            import demucs.hdemucs
            import numpy.core.multiarray
            import numpy
            import numpy.dtypes
            # Note: Pylance may report 'demucs.hdemucs' as unresolved, but it works at runtime if the UVR library is installed correctly.
            torch.serialization.add_safe_globals([demucs.hdemucs.HDemucs, numpy.core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float64DType])
            print("Added safe globals for UVR model loading (including NumPy globals) to handle PyTorch weights_only=True restriction.")
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not add safe globals for UVR model loading: {str(e)}. Attempting to load model without all necessary safe globals, which may fail due to PyTorch security restrictions.")
            print("Note: If the model fails to load due to 'weights_only' restrictions, additional globals may need to be allowlisted. Alternatively, you may manually modify the code to use 'weights_only=False' for torch.load() from trusted sources only.")
        
        try:
            # Attempt to load model on the selected device
            model = models.Demucs(name=model_name, other_metadata=model_config, device=device, logger=None)
            print(f"Model {model_name} loaded successfully on {device}.")
            return model
        except Exception as e:
            print(f"Failed to load model on {device} due to: {str(e)}. If this is an XPU compatibility issue (e.g., unsupported data types), attempting fallback to CPU.")
            if device == "xpu" and device_id == "xpu":
                device = "cpu"
                print(f"Falling back to load UVR model on CPU due to XPU device specified in CLI...")
                return models.Demucs(name=model_name, other_metadata=model_config, device=device, logger=None)
            else:
                raise
    except (ImportError, AttributeError) as e:
        print(f"Debug: Failed to import UVR library or access Demucs model: {str(e)}")
        raise ImportError("Ultimate Vocal Remover API library not detected or incompatible. Ensure 'ultimatevocalremover_api' is installed correctly (e.g., via 'pip install . ' from cloned repo 'https://github.com/NextAudioGen/ultimatevocalremover_api.git'). Attempting fallback to original audio.")

def apply_vocal_removal(audio: np.ndarray, sampling_rate: int, model: Optional[object] = None, 
                        model_name: str = "hdemucs_mmi", model_dir: str = "./uvr_models", device_id: str = "0") -> np.ndarray:
    """
    Apply vocal removal to audio, separating vocals from background music using ultimatevocalremover_api.
    
    Args:
        audio: Input audio array.
        sampling_rate: Sampling rate of the audio.
        model: Pre-loaded UVR model instance, if available.
        model_name: Name of the UVR model to use if model is not provided.
        model_dir: Directory to store downloaded models (not directly used but kept for compatibility).
        device_id: Device ID from CLI argument to determine if XPU is used for fallback logic.
    
    Returns:
        Processed audio array containing only vocals.
    """
    # Store original audio for fallback to avoid unnecessary conversions
    original_audio = audio.copy()
    
    if model is None:
        try:
            model = get_vocal_separator(model_name, model_dir, device_id)
        except Exception as e:
            print(f"Unable to load vocal removal model due to: {str(e)}. This could be due to a missing library, incompatible version, or unavailable model '{model_name}'. Ensure models are downloaded from 'https://github.com/NextAudioGen/ultimatevocalremover_api'. Proceeding with original audio as fallback.")
            print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
            return original_audio
    
    print("Separating vocals from background music...")
    print(f"Input audio stats - Length: {len(audio)} samples, Shape: {audio.shape}, Sample rate: {sampling_rate} Hz")
    
    if len(audio) == 0:
        print("Error: Input audio is empty. Cannot process empty audio. Proceeding with original audio as fallback.")
        return original_audio
    
    # Ensure audio is in stereo (2D) format as expected by UVR library
    if len(audio.shape) == 1:
        print(f"Converting mono audio to stereo by duplicating channel as required by UVR library.")
        audio = np.stack([audio, audio], axis=-1)
        print(f"Converted audio shape to {audio.shape}")
    elif len(audio.shape) > 2:
        print(f"Warning: Input audio has unexpected shape {audio.shape}. Expected 1D (mono) or 2D (stereo). Attempting to flatten or convert to suitable format.")
        if audio.shape[0] > 2:  # Likely (channels, samples)
            audio = audio.T  # Transpose to (samples, channels)
        audio = audio[:, :2] if audio.shape[-1] > 2 else audio  # Take first 2 channels if more are present
        print(f"Converted audio shape to {audio.shape}")
    elif audio.shape[-1] == 1:
        print(f"Converting mono audio to stereo by duplicating channel as required by UVR library.")
        audio = np.stack([audio[:, 0], audio[:, 0]], axis=-1)
        print(f"Converted audio shape to {audio.shape}")
    
    try:
        # Process audio using the UVR model
        print("Attempting audio processing with UVR model...")
        print(f"Audio data type: {audio.dtype}, Memory size: {audio.nbytes / 1024**2:.2f} MB")
        # Limit to first 30 seconds of audio for diagnostic purposes
        max_duration_s = 10
        max_samples = max_duration_s * sampling_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"Limited audio to first {max_duration_s} seconds for diagnostic testing. New length: {len(audio)} samples, New memory size: {audio.nbytes / 1024**2:.2f} MB")
        result = model(audio)
        if not isinstance(result, dict) or "separated" not in result:
            print("Error: Model output does not contain expected 'separated' key. Output structure invalid. Proceeding with original audio as fallback.")
            print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
            return original_audio
            
        separated_audio = result["separated"]
        if "vocals" in separated_audio:
            vocals = separated_audio["vocals"]
            print(f"Vocal separation successful. Vocals output shape: {vocals.shape}")
            # Transpose to ensure (samples, channels) format if needed
            if len(vocals.shape) == 2 and vocals.shape[0] < vocals.shape[1]:
                vocals = vocals.T
                print(f"Transposed vocals output to shape: {vocals.shape}")
            # Convert to mono if stereo for Whisper compatibility
            if len(vocals.shape) == 2 and vocals.shape[-1] == 2:
                vocals = np.mean(vocals, axis=-1)
                print(f"Converted vocals output to mono, final shape: {vocals.shape}")
        else:
            print("Warning: 'vocals' key not found in separated audio output. Available keys: " + str(list(separated_audio.keys())))
            print("Using original audio as fallback.")
            print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
            vocals = original_audio
    except Exception as e:
        print(f"Error processing audio with vocal removal model: {str(e)}. This might be due to model incompatibility, data format issues, or PyTorch weights loading restrictions. If the error mentions 'weights_only', consider manually allowlisting globals or loading with 'weights_only=False' only from trusted sources.")
        if device_id == "xpu":
            print("Since XPU device is specified, attempting fallback to CPU for processing...")
            try:
                model.to("cpu")
                print("Model moved to CPU for retry.")
                result = model(audio)
                if not isinstance(result, dict) or "separated" not in result:
                    print("Error on CPU retry: Model output does not contain expected 'separated' key. Output structure invalid. Proceeding with original audio as fallback.")
                    print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
                    return original_audio
                separated_audio = result["separated"]
                if "vocals" in separated_audio:
                    vocals = separated_audio["vocals"]
                    print(f"Vocal separation successful on CPU retry. Vocals output shape: {vocals.shape}")
                    # Transpose to ensure (samples, channels) format if needed
                    if len(vocals.shape) == 2 and vocals.shape[0] < vocals.shape[1]:
                        vocals = vocals.T
                        print(f"Transposed vocals output to shape: {vocals.shape}")
                    # Convert to mono if stereo for Whisper compatibility
                    if len(vocals.shape) == 2 and vocals.shape[-1] == 2:
                        vocals = np.mean(vocals, axis=-1)
                        print(f"Converted vocals output to mono, final shape: {vocals.shape}")
                else:
                    print("Warning on CPU retry: 'vocals' key not found in separated audio output. Available keys: " + str(list(separated_audio.keys())))
                    print("Using original audio as fallback.")
                    print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
                    vocals = original_audio
            except Exception as cpu_e:
                print(f"Error processing audio on CPU fallback: {str(cpu_e)}. Proceeding with original audio as fallback.")
                print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
                vocals = original_audio
        else:
            print("Proceeding with original audio as fallback.")
            print(f"Original audio stats - Length: {len(original_audio)} samples, Sample rate: {sampling_rate} Hz")
            vocals = original_audio
    
    return vocals
