# Qwen-Image-2.1, one block at a time

[Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) is three weight blocks
totalling **30.9 GB in bf16**:

| Block | Class | Params | bf16 weights |
|---|---|---:|---:|
| Text encoder | `Qwen3VLForConditionalGeneration` | 8.77 B | 16 722 MB |
| Transformer (DiT) | `QwenImage21Transformer2DModel` | 7.12 B | 13 571 MB |
| VAE | `AutoencoderKLQwenImage21` | 0.34 B | 644 MB |

That does not fit on a 24 GB card, and on a 32 GB card it fits with only ~600 MB
to spare — less than any single stage's activation working set, so it loads and
then OOMs on the first forward pass.

This runs the three blocks as **three separate processes** that hand each other
tensors on disk. Only one block is ever on the GPU, peak VRAM is **17.6 GB**, and
just **5.1 MB** of intermediate tensors crosses the process boundaries.

```
fetch ──► vl ──────────────► dit ──────────► vae ──► image.png
          Qwen3-VL           DiT, 40 steps   VAE
          16.7 GB            13.6 GB         0.6 GB
               prompt_embeds.pt    latents.pt
               [1,637,4096] bf16   [1,1024,64] bf16
               4.98 MB             0.127 MB
```

## Run it

```bash
export RUNPOD_KEY=...
python3 deploy.py run --gpu "NVIDIA GeForce RTX 3090" --out results_3090
```

One command: rents the pod, runs the pipeline, downloads the image, metrics and
log into `--out`, and terminates the pod. Other GPUs work the same way — pass
any `gpuTypeId` RunPod accepts (`NVIDIA GeForce RTX 5090`, `NVIDIA RTX A5000`, …).

Knobs, all environment variables read by `run_all.sh`: `HEIGHT`, `WIDTH` (must be
multiples of 32), `STEPS`, `SEED`. The prompt is `prompt.txt`.

### Other commands

| Command | What it does |
|---|---|
| `deploy.py create --gpu NAME` | rent a pod and start the run, leave it up |
| `deploy.py status` | pod state, cost so far, tail of the run log |
| `deploy.py log` | the whole run log |
| `deploy.py pull DIR` | download artifacts and metrics |
| `deploy.py kill` | terminate the pod in `.pod.json` |
| `deploy.py reap` | terminate **every** pod on the account |

`run` terminates the pod on every exit path, including Ctrl-C, via `atexit` plus
signal handlers. `reap` is the backstop if something still escapes.

## Files

| File | Role |
|---|---|
| `deploy.py` | RunPod lifecycle: rent, watch, collect, terminate. Runs on your machine. |
| `boot.sh` | Container entrypoint: file server, CUDA preflight, hand off to `run_all.sh`. |
| `run_all.sh` | Pod-side orchestrator: installs deps, runs each stage as its own process. |
| `run_stage.py` | The stages themselves, plus all instrumentation. |
| `prompt.txt` | The pre-expanded prompt. |
| `make_report.py` | Renders `metrics.json` into the report's tables. |
| `results*/` | Per-GPU output: image, `metrics.json`, `run.log`. |
| `REPORT.md` | Findings and numbers. |

## How the split works

The stages exploit two things about diffusers' `QwenImage21Pipeline`.

**Passing a component as `None` skips loading it.** `from_pretrained` filters the
init dict through a `load_module` check, so a component explicitly set to `None`
never has its shards read:

```python
# vl stage: text encoder only
pipe = QwenImage21Pipeline.from_pretrained(
    MODEL_DIR, transformer=None, vae=None, dtype=torch.bfloat16)
prompt_embeds, mask, _ = pipe.encode_prompt(prompt=prompt, device="cuda")

# dit stage: transformer only
pipe = QwenImage21Pipeline.from_pretrained(
    MODEL_DIR, text_encoder=None, vae=None, dtype=torch.bfloat16)
latents = pipe(prompt_embeds=prompt_embeds, prompt_embeds_mask=mask,
               height=512, width=512, num_inference_steps=40,
               output_type="latent").images
```

**`output_type="latent"` returns before the VAE is touched**, so stage 2 never
needs the decoder. Stage 3 loads `AutoencoderKLQwenImage21` on its own and
reproduces the pipeline's tail by hand: unpack the spatial flatten, undo the
latent normalisation with `latents_mean`/`latents_std`, decode, postprocess.

The processor (15 MB of tokenizer) loads in all three stages, because
`QwenImage21Pipeline.__init__` derives its prompt-template token offsets from it.

Each stage is a **separate process**, not just a separate `del`. That is what
guarantees the previous block is gone from both VRAM and host RAM — a CUDA
context does not fully release until the process exits.

## Pitfalls, and how they are handled

Five pods were burned getting the first run to work. Each failure now has a fix
in the code rather than a note in someone's head.

**1. `dockerArgs` must be a single line.** RunPod stores the container start
command verbatim. A value containing newlines leaves the container unable to
start, with no error anywhere — `runtime` stays `null` forever and the pod bills
the whole time. *Fix:* `deploy.py`'s `BOOT` is one line that clones the repo and
runs `boot.sh`; all real logic lives in the repo.

**2. CUDA forward compatibility bricks GeForce cards.** RunPod's PyTorch images
ship CUDA forward-compat libraries so a newer CUDA build can run on an older
driver. That is a datacenter-GPU feature. On a GeForce card,
`cudaGetDeviceCount` fails with **error 804**, `torch.cuda.is_available()`
returns `False`, and the pipeline silently runs on the CPU. *Fix:* pin a
`cu1281` image; `boot.sh` strips `/compat` from `LD_LIBRARY_PATH` and re-checks;
if the GPU is still unreachable it writes `FAILED:cuda` and `deploy.py` rotates
to another host. The `env` stage also aborts before the 32 GB download if torch
cannot see a GPU.

**3. Hosts that never finish pulling the image.** Two of five pods sat at
`runtime: null` for 15–30 minutes. *Fix:* `BOOT_TIMEOUT` (11 min) with automatic
rotation to another machine.

**4. Don't name a `threading.Event` `self._stop`.** `Thread.join` calls a private
`_stop()`; shadowing it makes every stage compute correctly and then die in
teardown with `'Event' object is not callable`.

**5. The download dominates.** 31.6 GB at 14.4 MB/s unauthenticated was 36 of the
43 minutes of the successful run. Set `HF_TOKEN` in the pod environment if you
have one; `HF_HUB_ENABLE_HF_TRANSFER=1` and `max_workers=16` are already on.

## Control plane

The pod's only outbound channel is `python3 -m http.server` serving
`/workspace/out` read-only through RunPod's HTTPS proxy at
`https://<pod-id>-8000.proxy.runpod.net`. `deploy.py` polls a `STATUS` file for
the current stage and downloads the results. There is deliberately no way to
send a command to the pod: everything it runs comes from a git commit, which
also means every run is reproducible from a ref.

RunPod's GraphQL API sits behind Cloudflare, which rejects the default
`urllib` User-Agent with error 1010 — hence the explicit `User-Agent` header in
`gql()`.

## Results

See [`REPORT.md`](REPORT.md).
