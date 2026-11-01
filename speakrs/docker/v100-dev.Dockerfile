# V100 (sm_70) kernel dev box: cudakit's GPU box toolchain (LLVM 21, CUDA 13.0 for
# cuda-oxide, rustup, cargo-oxide) on a CUDA 12.9 base. CUDA 13 ptxas no longer targets
# Volta and 12.9 is the last toolkit that does, so its ptxas builds the sm_70 cubins
#
# Build with cudakit's scripts directory as the context:
#   docker build -t speakrs-v100-dev -f speakrs/docker/v100-dev.Dockerfile cudakit/scripts
# Run it as a long-lived box. The GPU may be shared with other work, so keep VRAM use
# small and take timings under /workspace/gpu-bench.lock:
#   docker run -d --name speakrs-v100 --runtime nvidia --gpus all --cap-add SYS_ADMIN \
#     -v <src>:/src -v <models>:/workspace/models -v <target>:/cache/target speakrs-v100-dev
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG UID=1000
ARG GID=1000
# the host has 4 cores shared with other services, so leave one free for them
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA12_DOTTED=12.9 \
    CARGO_BUILD_JOBS=3

# Nsight Compute 2025.2 and the Nsight Systems of CUDA 12.9 are the last releases that
# profile Volta. Profiling needs root for the counters: `docker exec -u 0`
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nsight-compute-12-9 cuda-nsight-systems-12-9 python3 python3-venv \
        strace ripgrep unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

COPY setup-gpu-box.sh /opt/cudakit/setup-gpu-box.sh
RUN bash /opt/cudakit/setup-gpu-box.sh \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace/home /cache/target \
    && chown -R "${UID}:${GID}" /workspace /cache

# `bash -c` and login shells both see the toolchain variables
ENV BASH_ENV=/workspace/env.sh \
    HOME=/workspace/home \
    CARGO_TARGET_DIR=/cache/target
RUN echo '. /workspace/env.sh' >/etc/profile.d/10-workspace-env.sh

USER ${UID}:${GID}
WORKDIR /src
CMD ["sleep", "infinity"]
