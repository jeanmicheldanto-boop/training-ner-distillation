# Force new build (retry 2025-11-08)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy repo
COPY . /app

# Install Python packages
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r training_ner/requirements.txt && \
    python3.11 -m pip install --no-cache-dir runpod

# Run handler
CMD ["python3.11", "-m", "runpod.serverless.worker", "handler:handler"]
