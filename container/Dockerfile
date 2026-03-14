# NanoClaw Sandbox Container
# Minimal bash sandbox for agent tool isolation.
# The LLM (Ollama) runs on the host — this container only executes bash commands
# via `docker exec` to maintain filesystem and process isolation.

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    bash \
    curl \
    git \
    wget \
    python3 \
    python3-pip \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directories
RUN mkdir -p /workspace/group /workspace/global /workspace/ipc

# Non-root user for security (use -o to allow non-unique UID if 1000 is taken)
RUN useradd -m -u 1000 -o -s /bin/bash sandbox \
    && chown -R sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace/group

# Stay alive for docker exec calls from the host agent engine
CMD ["sleep", "infinity"]
