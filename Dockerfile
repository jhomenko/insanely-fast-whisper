# Dockerfile for insanely-fast-whisper with Intel XPU support
FROM ubuntu:24.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PATH=/usr/local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    python3.10 \
    python3.10-venv \
    python3-pip \
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
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy project files
WORKDIR /app
COPY . /app

# Install project dependencies from pyproject.toml and additional requirements for FastAPI server
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir intel-extension-for-pytorch \
    && pip install --no-cache-dir fastapi uvicorn python-dotenv

# Expose port for FastAPI server
EXPOSE 8000

# Default command to run the CLI application
CMD ["python3", "src/insanely_fast_whisper/cli.py", "--help"]
