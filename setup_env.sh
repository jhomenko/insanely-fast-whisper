#!/bin/bash
# Setup script for insanely-fast-whisper with Intel XPU support

echo "Setting up environment for insanely-fast-whisper with Intel XPU support..."

# Export the required environment variable for libzero linker
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Create or update the conda environment
echo "Creating/updating conda environment..."
conda env create -f env_full.yml --force

# Activate the environment
echo "Activating environment..."
conda activate IFW0701

# Install vocal removal dependencies with --no-deps
echo "Installing vocal removal dependencies..."
pip install git+https://github.com/jhj0517/ultimatevocalremover_api.git --no-deps
pip install git+https://github.com/jhj0517/pyrubberband.git --no-deps

echo "Environment setup complete!"
echo "To activate the environment and set required variables, run:"
echo "conda activate IFW0701"
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
