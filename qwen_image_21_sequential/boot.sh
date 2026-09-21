#!/usr/bin/env bash
# Container entrypoint, invoked by a deliberately trivial one-line `dockerArgs`
# (RunPod does not handle a multi-line container start command).
set -u

OUT=/workspace/out
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo boot > "$OUT/STATUS"
echo "[boot] $(date -u) starting"
nvidia-smi || true
df -h /workspace || true

# The file server is the only channel back out, so start it before anything
# that can fail -- otherwise a failure is invisible from outside the pod.
nohup python3 -m http.server 8000 --directory "$OUT" > /workspace/httpd.log 2>&1 &
sleep 2
echo "[boot] file server up on :8000"

# --- CUDA preflight -------------------------------------------------------
# A cu12.x torch build runs on any driver from 525 up, but only when it links
# the host driver. RunPod's PyTorch images also ship CUDA forward-compatibility
# libraries, and when those land on the loader path a GeForce card fails with
# `cudaGetDeviceCount` error 804: torch reports no GPU and the whole pipeline
# silently runs on the CPU. Strip them and re-check before committing to a
# 32 GB download.
cuda_ok() { python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if ! cuda_ok; then
    echo "[boot] torch cannot open the GPU; dropping CUDA compat libs from LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd:)"
    if ! cuda_ok; then
        echo "[boot] GPU still unreachable on this host -- see the error above"
        python3 -c "import torch; torch.cuda.init()" || true
        echo "FAILED:cuda" > "$OUT/STATUS"
        sleep infinity
    fi
    echo "[boot] recovered: GPU visible after dropping compat libs"
fi
python3 - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"[boot] {p.name}  {p.total_memory / 1024**3:.2f} GiB  sm_{p.major}{p.minor}  torch {torch.__version__}")
PY

bash /workspace/repo/qwen_image_21_sequential/run_all.sh
echo "[boot] run_all.sh exited rc=$?"
# Stay up so the results stay downloadable; deploy.py terminates the pod.
sleep infinity
