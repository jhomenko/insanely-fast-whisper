#!/bin/bash
# Installation script for vocal removal dependencies

echo "Installing vocal removal dependencies..."
echo "This script installs Ultimate Vocal Remover API and pyrubberband with --no-deps flag"
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment detected. Please activate your environment first:"
    echo "conda activate IFW0701"
    echo "or"
    echo "conda activate insanely-fast-whisper-intel"
    echo ""
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Install Ultimate Vocal Remover API
echo "Installing Ultimate Vocal Remover API..."
pip install git+https://github.com/jhj0517/ultimatevocalremover_api.git --no-deps

if [ $? -eq 0 ]; then
    echo "✅ Ultimate Vocal Remover API installed successfully"
else
    echo "❌ Failed to install Ultimate Vocal Remover API"
    exit 1
fi

# Install pyrubberband
echo "Installing pyrubberband..."
pip install git+https://github.com/jhj0517/pyrubberband.git --no-deps

if [ $? -eq 0 ]; then
    echo "✅ pyrubberband installed successfully"
else
    echo "❌ Failed to install pyrubberband"
    exit 1
fi

echo ""
echo "🎉 Vocal removal dependencies installed successfully!"
echo ""
echo "You can now use vocal removal with:"
echo "python -m src.insanely_fast_whisper.cli --file-name input/your_audio.m4a --vocal-removal True --vocal-method uvr"
echo ""
echo "Available vocal removal methods:"
echo "  --vocal-method uvr     : Ultimate Vocal Remover (default)"
echo "  --vocal-method hdemucs : Hybrid Demucs"
echo ""
echo "For UVR method, you can also specify:"
echo "  --vocal-model UVR-MDX-NET-Inst_HQ_4  : Model name (default)"
echo "  --vocal-model-dir ./uvr_models        : Model directory (default)"
