#!/usr/bin/env bash
# Pod-side orchestrator. Runs each stage as a separate process so that the
# previous stage's weights are gone from both host RAM and VRAM before the
# next one loads.
set -u

OUT=/workspace/out
ART=$OUT/artifacts
mkdir -p "$ART" "$OUT/metrics"

# boot.sh resolves the interpreter that owns torch; fall back for a direct run.
PY=${PY:-python3}

export MODEL_DIR=/workspace/models/Qwen-Image-2.1
export ART_DIR=$ART
export METRICS_DIR=$OUT/metrics
export PROMPT_FILE=/workspace/repo/qwen_image_21_sequential/prompt.txt
export HEIGHT=${HEIGHT:-512}
export WIDTH=${WIDTH:-512}
export STEPS=${STEPS:-40}
export SEED=${SEED:-42}
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

STAGE_PY=/workspace/repo/qwen_image_21_sequential/run_stage.py

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

status() { echo "$1" > "$OUT/STATUS"; }

status "installing"
say "installing dependencies into $("$PY" -c 'import sys; print(sys.executable)')"

# Debian-packaged interpreters mark themselves externally managed (PEP 668) and
# refuse a plain `pip install`. This is a single-purpose container, so installing
# into the system interpreter is the point -- but only pass the flag if this pip
# understands it.
PIP_FLAGS=""
if "$PY" -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
    PIP_FLAGS="--break-system-packages"
    say "pip: using --break-system-packages (PEP 668 interpreter)"
fi

pip_install() {
    # A bare `pip ... | tail` reports tail's exit status, which hides every
    # install failure until something downstream fails to import.
    if ! "$PY" -m pip install --no-cache-dir -q $PIP_FLAGS "$@" > /tmp/pip.log 2>&1; then
        say "pip install failed:"
        tail -30 /tmp/pip.log
        status "FAILED:install"
        exit 1
    fi
}

pip_install -U "transformers>=5.17" accelerate safetensors pillow \
    huggingface_hub hf_transfer importlib_metadata filelock regex requests numpy
# --no-deps keeps pip from touching the image's torch build.
pip_install -U --no-deps "git+https://github.com/huggingface/diffusers"

if ! "$PY" -c "import diffusers, transformers, huggingface_hub, accelerate" 2>&1; then
    say "dependencies did not import after install"
    status "FAILED:install"
    exit 1
fi
say "dependencies installed"

run_stage() {
    local name=$1
    status "$name"
    say "---------- stage $name ----------"
    local t0=$SECONDS
    "$PY" "$STAGE_PY" "$name"
    local rc=$?
    say "stage $name finished rc=$rc in $((SECONDS - t0))s"
    if [ $rc -ne 0 ] && [ "$name" != "fit" ]; then
        status "FAILED:$name"
        say "ABORTING"
        exit $rc
    fi
}

run_stage env
run_stage fetch
run_stage vl
run_stage dit
run_stage vae
# Control run, last: shows whether the three blocks co-reside on this card.
run_stage fit

say "collecting metrics"
"$PY" - <<'COLLECT'
import glob, json, os
out = {}
for p in sorted(glob.glob("/workspace/out/metrics/*.json")):
    out[os.path.basename(p)[:-5]] = json.load(open(p))
json.dump(out, open("/workspace/out/metrics.json", "w"), indent=2)
print(json.dumps({k: v.get("latency_s", {}).get("total") for k, v in out.items()}, indent=2))
COLLECT

ls -l "$ART"
status "DONE"
say "ALL DONE"
