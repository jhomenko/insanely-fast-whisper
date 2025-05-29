import whisper
import torch
import time

# --- Configuration ---
AUDIO_FILE_PATH = "/tmp/GreenParty.wav"
# You can choose different model sizes: 'tiny', 'base', 'small', 'medium', 'large'
MODEL_SIZE = "medium" # Using medium for comparison

# --- Device Setup ---
device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

# Check if XPU is available and set the device for torch
if device.type == "xpu":
    torch.xpu.set_device(device) # Explicitly set the current device for torch
    # Enable static cache and potentially compile the forward pass (details below)
    # This might require specific configurations or environment variables for XPU

# --- Load Model ---
print(f"Loading Whisper model: {MODEL_SIZE}")
# openai-whisper loads the model to the default device automatically
model = whisper.load_model(MODEL_SIZE)
print("Model loaded.")

# Move model to the desired device if not already there
model = model.to(device)
print(f"Model moved to device: {model.device}")

# --- Apply torch.compile ---
if device.type == "xpu":
    print("Applying torch.compile to the model's forward pass...")
    try:
        # You might need to experiment with different modes
        # "reduce-overhead" or "max-autotune" are common choices
        # fullgraph=True might be too strict if chunking involves dynamic graphs
        model.forward = torch.compile(model.forward, mode="reduce-overhead")
        print("torch.compile applied successfully.")
    except Exception as e:
        print(f"Failed to compile the model: {e}")
        print("Proceeding without torch.compile.")

# --- Transcribe ---
print(f"Transcribing: {AUDIO_FILE_PATH}")
start_time = time.time()

try:
    # openai-whisper's transcribe method handles the process
    # It internally uses the model's forward pass, which is now compiled
    result = model.transcribe(AUDIO_FILE_PATH)

    end_time = time.time()
    transcription_time = end_time - start_time
    print("Transcription complete!")
    print(f"Transcription time: {transcription_time:.2f} seconds")

    # Print a snippet of the output
    if result and "text" in result:
        print("\n--- Transcription Snippet ---")
        print(result["text"][:500] + "...") # Print first 500 characters

except Exception as e:
    print(f"\nAn error occurred during transcription: {e}")
