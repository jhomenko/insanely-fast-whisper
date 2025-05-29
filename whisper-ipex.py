import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import is_flash_attn_2_available

torch.Tensor.double = torch.Tensor.float
torch.float64 = torch.double = torch.float32

# Import the specific functions from the utils.py file
# Assuming you have the utils.py file saved locally, you can import it directly
# Make sure the path below is correct
import sys
sys.path.append("/mnt/data/projects/insanely-fast-whisper/whisper-ipex.py") # Add the directory to your path
from utils import whisper_generate # Import the function

# Make sure your Conda environment is active and oneAPI setvars.sh is sourced

# 1. Load the Whisper model and processor
model_name = "openai/whisper-medium" # Or another Whisper model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Move model to the appropriate device (XPU)
device = "xpu:0" if torch.xpu.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Apply IPEX optimizations
# The exact IPEX optimization steps might vary depending on your IPEX version
# and the specific recommendations for Whisper.
# A common pattern is using ipex.optimize:

import intel_extension_for_pytorch as ipex

model = ipex.optimize(model.eval(), dtype=torch.float32 if device != "cpu" else torch.float32) # Optimize for inference
model.to(torch.float32)

# 2. Prepare audio data
audio_file_path = "/tmp/GreenParty.wav" # Replace with your audio file

# Load audio and resample if necessary
import librosa
import soundfile as sf

try:
    audio, sample_rate = sf.read(audio_file_path)
except ImportError:
    print("Please install soundfile: pip install soundfile")
    exit()

if sample_rate != 16000:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    sample_rate = 16000

# Convert to torch tensor and move to device
input_audio = torch.tensor(audio) # Keep it on CPU initially

# Prepare input features using the processor
# Pass the CPU tensor to the processor
input_features = processor(
    input_audio,
    sampling_rate=sample_rate,
    return_tensors="pt",
    padding="longest",
    return_attention_mask=True).input_features

# Move the resulting input_features to the XPU device AND convert to float16
input_features = input_features.to(device, dtype=torch.float32)


# 3. Use the optimized whisper_generate function
# We'll pass the model object as the first argument to the function
# The function is likely designed to be a standalone utility that works with the model object
# and the input features.
# You might need to pass other arguments based on the function's requirements and your desired output
# (e.g., return_token_timestamps=True, return_segments=True, language="en", task="transcribe")

# Let's start with a basic call, assuming the function handles default parameters
# If you need specific language, task, or timestamp output, you can add those arguments
try:
    # The whisper_generate function in the utils.py takes 'self' as the first argument,
    # which suggests it's intended to be a method of a class.
    # We need to call it as if it were a method of our 'model' object.
    # We can use types.MethodType to dynamically add it as a method for this instance.
    import types
    model.whisper_generate = types.MethodType(whisper_generate, model)


    print(f"Model dtype before generate: {model.dtype}")
    print(f"Input features dtype before generate: {input_features.dtype}")
    print(f"Input features device before generate: {input_features.device}")

    # Now call the method on the model instance
    result = model.whisper_generate(input_features=input_features, 
        return_segments=True,  
        num_segment_frames=16000 * 30, 
        return_timestamps=True)

    # 4. Process the output
# The structure of the 'result' is a dictionary with a 'segments' key
# when return_segments=True is used in whisper_generate
    if isinstance(result, dict) and "segments" in result:
        print("Transcription:")
    # Iterate directly through the list of segment dictionaries
        for segment in result["segments"]:
        # Access the timestamp and text keys directly
            if isinstance(segment, dict) and 'timestamp' in segment and 'text' in segment:
                start_time = segment["timestamp"][0]
                end_time = segment["timestamp"][1]
                text = segment["text"]
                print(f"[{start_time:.2f} - {end_time:.2f}] {text}")

# You can keep the other `elif` and `else` blocks if you anticipate
# the whisper_generate function returning other formats in different scenarios,
# but for your current goal of processing long audio with segments,
# the main `if` block is what you need.
#
# For clarity and focusing on the segment output, you could even simplify it further:
# if isinstance(result, dict) and "segments" in result:
#     print("Transcription:")
#     for segment in result["segments"]:
#         start_time = segment["timestamp"][0]
#         end_time = segment["timestamp"][1]
#         text = segment["text"]
#         print(f"[{start_time:.2f} - {end_time:.2f}] {text}")
# else:
#     print("Unexpected output format from whisper_generate. Expected dictionary with 'segments'.")
#     print(result)


except AttributeError:
    print("Could not dynamically add whisper_generate as a method. Ensure 'types' is imported.")
    print("Alternatively, the whisper_generate function might need to be called differently based on the IPEX example.")
except Exception as e:
    print(f"An error occurred during transcription: {e}")

