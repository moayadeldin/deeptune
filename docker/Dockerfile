
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install common system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA-enabled PyTorch.
RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    --only-binary=:all: \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy the package source.
COPY . .

# Install DeepTune and all declared dependencies.
RUN python -m pip install .

# Verify that PyTorch remains CUDA-enabled
# after installing the project.
RUN python -m pip check && \
    python -c "import torch; assert torch.version.cuda is not None, 'CUDA-enabled PyTorch is required'; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "deeptune"]

CMD ["--help"]
