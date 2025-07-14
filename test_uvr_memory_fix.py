#!/usr/bin/env python3
"""
Test script to verify the UVR memory leak fix.
"""
import numpy as np
import time
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_uvr_memory_leak():
    """Test UVR implementation for memory leaks"""
    
    print("Testing UVR Memory Leak Fix")
    print("=" * 50)
    
    # Import after checking memory baseline
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    try:
        from src.insanely_fast_whisper.utils.uvr import apply_vocal_removal
        
        # Create test audio (10 seconds of random noise)
        sample_rate = 16000
        duration = 10  # seconds
        test_audio = np.random.randn(sample_rate * duration).astype(np.float32)
        
        print(f"Test audio created: {test_audio.shape}, duration: {duration}s")
        
        # Test multiple runs to check for memory leaks
        for i in range(3):
            print(f"\nRun {i+1}:")
            
            before_memory = get_memory_usage()
            print(f"  Memory before: {before_memory:.2f} MB")
            
            start_time = time.time()
            result = apply_vocal_removal(test_audio, sample_rate, device_id="cpu")
            end_time = time.time()
            
            after_memory = get_memory_usage()
            print(f"  Memory after: {after_memory:.2f} MB")
            print(f"  Memory delta: {after_memory - before_memory:.2f} MB")
            print(f"  Processing time: {end_time - start_time:.2f} seconds")
            print(f"  Result shape: {result.shape}")
            
            # Wait a bit for garbage collection
            time.sleep(2)
        
        final_memory = get_memory_usage()
        total_memory_increase = final_memory - initial_memory
        
        print(f"\nFinal Results:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Total increase: {total_memory_increase:.2f} MB")
        
        if total_memory_increase < 100:  # Less than 100MB increase is acceptable
            print("✅ Memory leak test PASSED")
        else:
            print("❌ Memory leak test FAILED - excessive memory usage")
            
    except ImportError as e:
        print(f"❌ UVR library not installed: {e}")
        print("Please run: ./install_vocal_removal.sh")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_uvr_memory_leak()
