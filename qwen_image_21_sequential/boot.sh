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

fail() { echo "[boot] $2"; echo "$1" > "$OUT/STATUS"; sleep infinity; }

# --- find the interpreter that owns torch --------------------------------
# Images disagree about where that is: RunPod's is the system python3, the
# upstream pytorch/pytorch images put it under /opt/conda and do not always
# add it to PATH. Resolve it once and let everything downstream inherit it.
PY=""
for cand in python3 /opt/conda/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch" >/dev/null 2>&1; then
        PY="$cand"; break
    fi
done
[ -n "$PY" ] || fail "FAILED:cuda" "no python on this image can import torch"
export PY
echo "[boot] python: $("$PY" -c 'import sys, torch; print(sys.executable, "torch", torch.__version__)')"

# The file server is the only channel back out, so start it before anything
# else that can fail -- otherwise a failure is invisible from outside the pod.
nohup "$PY" -m http.server 8000 --directory "$OUT" > /workspace/httpd.log 2>&1 &
sleep 2
echo "[boot] file server up on :8000"

# --- CUDA preflight -------------------------------------------------------
# A cu12.x torch build runs on any driver from 525 up, but only when it links
# the host driver. Some images also ship CUDA forward-compatibility libraries,
# and when those land on the loader path a GeForce card fails with
# `cudaGetDeviceCount` error 804: torch reports no GPU and the whole pipeline
# silently runs on the CPU. Strip them and re-check before committing to a
# 32 GB download.
cuda_ok() { "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if ! cuda_ok; then
    echo "[boot] torch cannot open the GPU; dropping CUDA compat libs from LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd:)"
    if ! cuda_ok; then
        "$PY" -c "import torch; torch.cuda.init()" || true
        fail "FAILED:cuda" "GPU still unreachable on this host -- see the error above"
    fi
    echo "[boot] recovered: GPU visible after dropping compat libs"
fi
"$PY" - <<'PREFLIGHT'
import torch
p = torch.cuda.get_device_properties(0)
print(f"[boot] {p.name}  {p.total_memory / 1024**3:.2f} GiB  sm_{p.major}{p.minor}")
PREFLIGHT

bash /workspace/repo/qwen_image_21_sequential/run_all.sh
echo "[boot] run_all.sh exited rc=$?"
# Stay up so the results stay downloadable; deploy.py terminates the pod.
sleep infinity
