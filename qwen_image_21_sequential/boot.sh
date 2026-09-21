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

# Read-only static server, so results can be collected over RunPod's HTTPS proxy.
nohup python3 -m http.server 8000 --directory /workspace/out > /workspace/httpd.log 2>&1 &
sleep 2
echo "[boot] file server up on :8000"

bash /workspace/repo/qwen_image_21_sequential/run_all.sh
echo "[boot] run_all.sh exited rc=$?"
sleep infinity
