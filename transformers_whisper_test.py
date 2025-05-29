import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
import time
import soundfile as sf # Using soundfile to load the .wav file
import numpy as np

# --- Configuration ---
AUDIO_FILE_PATH = "/tmp/GreenParty.wav"
MODEL_NAME = "openai/whisper-medium" # Using medium model
BATCH_SIZE = 1 # Start with batch size 1
SAMPLING_RATE = 16000 # Whisper expects 16kHz audio

# --- Device Setup ---
# Use torch.device("xpu:0") to specify the first XPU device
device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "xpu":
    try:
        torch.xpu.set_device(device.index) # Use device.index to get the integer index
        print(f"XPU device set to: {torch.xpu.current_device()}")
    except Exception as e:
        print(f"Failed to set XPU device: {e}")
        print("Proceeding with potentially default device.")

# --- Load Processor and Model ---
print(f"Loading processor and model: {MODEL_NAME}")
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, # Use float16 for potentially better XPU performance
        low_cpu_mem_usage=True # Helpful for memory
    ).to(device)
    print("Processor and model loaded.")
except Exception as e:
    print(f"Failed to load processor or model: {e}")
    exit() # Exit if model loading fails

# --- Load and Process Audio ---
print(f"Loading audio: {AUDIO_FILE_PATH}")
try:
    audio_input, sampling_rate = sf.read(AUDIO_FILE_PATH)
    if sampling_rate != SAMPLING_RATE:
        print(f"Warning: Audio sampling rate is {sampling_rate}Hz, expected {SAMPLING_RATE}Hz. Resampling...")
        # Basic resampling (you might need a more robust library like torchaudio for better quality)
        # For simplicity, let's assume soundfile can handle basic resampling or that the audio is already 16kHz
        # If not, you'll need to add proper resampling code here.
        # For now, we'll proceed but be aware of potential quality issues if resampling is needed.
        # A proper implementation would use something like:
        # import torchaudio
        # waveform, sr = torchaudio.load(AUDIO_FILE_PATH)
        # if sr != SAMPLING_RATE:
        #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)
        #     waveform = resampler(waveform)
        # audio_input = waveform.squeeze().numpy()
        pass # Placeholder for resampling if needed

    # Process audio into input_features using the processor
    # Need to handle long audio: load the full audio, set truncation=False, padding="longest"
    # and return_attention_mask=True
    inputs = processor(
        audio_input,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        sampling_rate=SAMPLING_RATE
    ).to(device, torch.float16) # Move inputs to device and use float16

    print("Audio loaded and processed.")
    print(f"Input features shape: {inputs.input_features.shape}")
    print(f"Attention mask shape: {inputs.attention_mask.shape}")

except Exception as e:
    print(f"Failed to load or process audio: {e}")
    exit() # Exit if audio processing fails

# --- Transcribe using model.generate() ---
print("Starting transcription using model.generate()...")
start_time = time.time()

try:
    # Use model.generate() for long-form transcription
    # Need to pass input_features, attention_mask (if batch_size > 1),
    # and set return_timestamps=True
    # Also, consider other generation parameters like num_beams, temperature, etc.

    # Note: The transformers example used generate(**inputs). This passes all items in the 'inputs' dictionary
    # as keyword arguments to generate. Since we used return_attention_mask=True and padding="longest",
    # the 'inputs' dictionary will contain 'input_features' and 'attention_mask'.
    # If BATCH_SIZE > 1, attention_mask is needed. With BATCH_SIZE = 1, it might still be passed but less critical.
    # We also need to set return_timestamps=True for long-form.

    generated_ids = model.generate(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask, # Pass attention mask
        return_timestamps=True, # Essential for long-form transcription
        # Add other generation parameters here if needed, e.g.:
        # max_new_tokens=1024,
        # num_beams=5,
    )

    end_time = time.time()
    transcription_time = end_time - start_time
    print("Transcription complete!")
    print(f"Transcription time: {transcription_time:.2f} seconds")

    # Decode the generated ids back to text using the processor
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Print the transcription
    print("\n--- Full Transcription ---")
    print(transcription[0]) # For a single audio file

except Exception as e:
    print(f"\nAn error occurred during transcription: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback
