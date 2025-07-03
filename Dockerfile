# Dockerfile for insanely-fast-whisper with Intel XPU support
FROM ubuntu:24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PATH=/usr/local/bin:$PATH \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Intel graphics packages for XPU support as per Intel's lightweight approach
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo \
    && apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo \
    && apt-get install -y libze-dev intel-ocloc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add user to render group for access to /dev/dri/renderD*
RUN groupadd -f render && usermod -aG render root

# Set up Python environment
RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy project files
WORKDIR /app
COPY . /app

# Install project dependencies from pyproject.toml and additional requirements for FastAPI server
# Use specific index URLs and versions for PyTorch and Intel Extension for PyTorch for XPU support
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/xpu \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    && pip install --no-cache-dir --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
    intel-extension-for-pytorch==2.7.10+xpu \
    oneccl_bind_pt==2.7.0+xpu \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir fastapi uvicorn python-dotenv python-multipart

# Copy the patched transformers library to overwrite the default installation
COPY transformers_Patched /opt/venv/lib/python3.11/site-packages/transformers

# Expose port for FastAPI server
EXPOSE 8000

# Default command to run the FastAPI server with preconfigured arguments for XPU support and optimizations
# Also, verify the LD_PRELOAD library path or find the correct one dynamically
CMD ["bash", "-c", "if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6; else echo 'LD_PRELOAD path not found, searching for libstdc++.so.6...'; export LD_PRELOAD=$(find /usr/lib* -name 'libstdc++.so.6' -print -quit); fi && echo 'LD_PRELOAD set to: $LD_PRELOAD' && python3 server.py --host 0.0.0.0 --port 8000 --device-id xpu --batch-size 6 --vad-filter True"]
