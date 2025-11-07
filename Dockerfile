FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy repo
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir -r training_ner/requirements.txt

# Install RunPod handler
RUN pip install --no-cache-dir runpod

# Run handler
CMD ["python", "-m", "runpod.serverless.worker", "handler:handler"]
