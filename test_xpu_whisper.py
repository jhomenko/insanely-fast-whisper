import torch
import intel_extension_for_pytorch as ipex
from transformers import WhisperProcessor, AutoModelForSpeechSeq2Seq
import librosa
import time

def test_xpu_whisper():
    print("Testing XPU functionality with Whisper model...")

    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"{'XPU device detected.' if device == 'xpu' else 'Falling back to CPU.'}")

    model_name = "openai/whisper-small"
    print(f"Loading model: {model_name}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_cache=True
    )

    # Set model to eval mode BEFORE optimization (per IPEX recommendation)
    model.eval()
    
    # Optimize with IPEX (llm-specific API for transformer models)
    print("Optimizing model with ipex.llm...")
    model = ipex.llm.optimize(model, dtype=torch.float16, device=device, low_bit="bf16")

    # Move model to device after optimization
    model.to(device)
    print(f"Model optimized and moved to device: {device}")

    processor = WhisperProcessor.from_pretrained(model_name)
    audio_path = "Intro.m4a"
    print(f"Loading audio file: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, duration=10)
    print(f"Audio loaded, length: {len(audio)/sr:.2f} seconds")

    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device, dtype=torch.float16)
    print(f"Input features ready on device: {device}")

    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language='en', task='transcribe')
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    elapsed_time = time.time() - start_time

    print(f"Inference completed in {elapsed_time:.2f} seconds")
    print(f"Transcription result (preview): {transcription[:100]}...")
    return transcription

if __name__ == "__main__":
    try:
        test_xpu_whisper()
        print("Test completed successfully.")
    except Exception as e:
        print(f"Error during test: {str(e)}")
