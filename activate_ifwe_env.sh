#!/bin/bash
# Activation script for IFW0701 environment with Intel XPU support

echo "Activating IFW0701 environment with Intel XPU support..."

# Activate the conda environment
conda activate IFW0701

# Set the required environment variable for libzero linker
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "Environment activated successfully!"
echo "LD_PRELOAD set to: $LD_PRELOAD"
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo ""
echo "You can now run the whisper CLI with Intel XPU support:"
echo "python -m src.insanely_fast_whisper.cli --file-name input/your_audio.m4a --device-id xpu"
