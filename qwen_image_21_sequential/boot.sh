#!/usr/bin/env bash
# Container entrypoint, invoked by a deliberately trivial one-line
# `dockerArgs` (RunPod does not handle a multi-line container start command).
set -u
mkdir -p /workspace/out
exec > >(tee -a /workspace/out/run.log) 2>&1
echo boot > /workspace/out/STATUS
echo "[boot] $(date -u) starting"
nvidia-smi || true
df -h /workspace || true

# Safety net for the CUDA forward-compatibility trap: if the image ships
# compat libs that this (GeForce) card cannot use, drop them from the loader
# path so torch links against the host driver instead.
if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[boot] torch cannot see the GPU; dropping CUDA compat libs from LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd:)"
    python3 -c "import torch; print('[boot] cuda available after fix:', torch.cuda.is_available())" || true
fi

# Read-only static server, so results can be collected over RunPod's HTTPS proxy.
nohup python3 -m http.server 8000 --directory /workspace/out > /workspace/httpd.log 2>&1 &
sleep 2
echo "[boot] file server up on :8000"

bash /workspace/repo/qwen_image_21_sequential/run_all.sh
echo "[boot] run_all.sh exited rc=$?"
sleep infinity
