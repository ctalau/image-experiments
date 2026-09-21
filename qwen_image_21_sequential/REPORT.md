# Qwen-Image-2.1 on one consumer GPU, run stage by stage

Running [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) text-to-image on a
single RunPod GPU, with the three weight blocks — the Qwen3-VL text encoder, the DiT,
and the VAE — executed as **three separate processes** that hand each other tensors on
disk, so only one block is on the GPU at a time.

Target: a 512×512 image from a pre-expanded prompt for *"swiss army knife penguin"*.

Two runs: an **RTX 5090 (32 GB)** first, then an **RTX 3090 (24 GB)** on the community
cloud, where the split stops being an optimisation and becomes the only way to run the
model at all. The 3090 run is in [`results_3090/`](results_3090/), the 5090 run in
[`results/`](results/); [`README.md`](README.md) documents the tooling.

## The image

![Swiss Army knife penguin, 512x512, Qwen-Image-2.1](results/image_rgb.png)

*512×512, 40 steps, seed 42, no classifier-free guidance (Qwen-Image-2.1 is meant to be
sampled at `true_cfg_scale=1.0`). The model decoded it natively as RGBA — the copy above
is flattened onto white; `results/image.png` is the original 4-channel output.*

The prompt fixed the crimson anodised case, the cross-and-shield emblem, the five fanned
implements (blade, screwdriver, corkscrew, scissors, can opener), the flipper-as-implement
on the left, the orange webbed feet, and the closed pocket knife in the lower-left as a
scale reference. All of those survive into the render. The one clear miss is the implement
count on the right: the prompt asked for five and named each, and the image shows roughly
that many but merges the screwdriver and can opener into indistinct steel.

## Why run it sequentially

| Block | Class | Params | bf16 weights |
|---|---|---:|---:|
| Text encoder | `Qwen3VLForConditionalGeneration` | 8.77 B | 16 722 MB |
| Transformer (DiT) | `QwenImage21Transformer2DModel` | 7.12 B | 13 571 MB |
| VAE | `AutoencoderKLQwenImage21` | 0.34 B | 644 MB |
| **Total** | | **16.22 B** | **30 937 MB** |

On a 24 GB card that is simply too much. On a 32 GB card it fits, with 1 172 MB to
spare against 32 109 MB usable — and that is the whole problem: the two control runs
below show it loading and then having nowhere to compute.

## The RTX 5090 (32 GB)

### Environment

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 607 MiB installed / 32 109 MiB usable, sm_120 |
| Driver | 580.159.04 |
| Host | 236 vCPU, 1004 GB RAM (RunPod on-demand, $0.99/hr) |
| Container | `runpod/pytorch:1.3.2-cu1281-torch2130-ubuntu2404` |
| torch / CUDA | 2.13.0+cu129 / 12.9 |
| diffusers / transformers / accelerate | 0.41.0.dev0 (git main) / 5.17.0 / 1.15.0 |

### Checkpoint

| Folder | On disk |
|---|---:|
| `text_encoder/` | 16 722 MB |
| `transformer/` | 13 571 MB |
| `vae/` | 1 288 MB |
| `processor/` | 15 MB |
| **Total** | **31 597 MB** |

Download took **2 190 s (36.5 min) at 14.4 MB/s** — unauthenticated Hugging Face
downloads on this host, and by far the largest single cost in the experiment. Note the
VAE: 1 288 MB on disk but 644 MB in memory, because that checkpoint is stored fp32 and
cast to bf16 at load.

### Latency per stage (seconds)

| Stage | python import | load weights (CPU) | host → GPU | compute | save | **total** |
|---|---:|---:|---:|---:|---:|---:|
| `vl` — encode prompt | 11.09 | 4.20 | 8.02 | **2.80** | 0.01 | 26.11 |
| `dit` — 40 denoise steps | 11.49 | 3.28 | 7.67 | **5.85** | 0.00 | 28.71 |
| `vae` — decode | 16.66 | 11.06 | 1.06 | **1.25** | 0.51 | 30.67 |
| | | | | | | **85.49** |

**Actual model compute is 9.89 s of the 85.5 s.** The rest is overhead that the
sequential split imposes: 39.2 s of Python/torch/diffusers import across three fresh
interpreters, 18.5 s of reading safetensors, and 16.8 s of host→device copies. Measured
transfer rates were 2.04 GB/s for the text encoder and 1.73 GB/s for the DiT — pageable,
non-pinned copies, so well under what PCIe 5 could do.

Per-step DiT timings (40 steps): first step **1.043 s**, every other step a median of
**107 ms** (min 99 ms, max 200 ms). The first step is ~10× the rest because it prefills
the text KV cache — Qwen-Image-2.1's `causal_condition` makes the text keys and values
step-independent, so they are computed once and reused for the remaining 39 steps.

### Memory per stage (MB)

| Stage | weights | torch peak alloc | torch peak reserved | peak device used | activation overhead | peak host RSS |
|---|---:|---:|---:|---:|---:|---:|
| `vl` | 16 722 | 17 265 | 17 410 | 17 579 | 857 | 5 563 |
| `dit` | 13 571 | 14 172 | 14 254 | 14 855 | 1 284 | 10 332 |
| `vae` | 644 | 2 332 | 2 758 | 3 359 | 2 715 | 2 750 |

"Peak device used" is whole-device occupancy sampled at 5 Hz, so it includes the ~300 MB
CUDA context that `torch.cuda.max_memory_allocated` does not see. "Activation overhead"
is that figure minus the weights.

The worst stage peaks at **17.6 GB — 55% of the card**. Host RAM never exceeded 10.3 GB
per process, so the split costs nothing in system memory; the 1 TB on this host is
irrelevant, and the same run would fit comfortably in 16 GB of RAM.

The VAE is the interesting one: 644 MB of weights but 2 715 MB of activations, a 4.2×
ratio, because the decoder expands a 32×32×64 latent to 512×512×4 through five upsampling
levels. At 2K, the resolution the model is actually designed for, that term grows with
pixel count and becomes the binding constraint rather than the DiT.

### Control: can all three co-reside?

A fourth process loaded the whole pipeline onto the GPU at once:

| Metric | Value |
|---|---|
| Load to CPU | 21.1 s |
| Host → GPU | 26.3 s |
| torch peak allocated | 30 954 MB |
| **Peak device used** | **31 491 MB of 32 109 MB** |
| Free headroom | **618 MB** |
| OOM during load? | No |

So the monolithic pipeline *loads*. It just cannot **run**: 618 MB of headroom is less
than the activation working set of any single stage — 857 MB for the encoder, 1 284 MB
for the DiT, 2 715 MB for the VAE. A one-process run on this card would load fine and
then OOM on the first forward pass, which is the more annoying failure mode. Note also
that `.to("cuda")` took 26.3 s here versus 8.0 + 7.7 + 1.1 = 16.8 s across the three
stages: the allocator slows down markedly as the card fills.

`enable_model_cpu_offload()` is the supported alternative and would also work, but it
keeps one Python process alive holding all three blocks in host RAM and pays a
host→device transfer per block *per call*. The process split pays those transfers once
and keeps peak host RSS at 10 GB.

### Intermediate artifacts

| Artifact | Shape | dtype | In memory | On disk |
|---|---|---|---:|---:|
| `prompt_embeds.pt` | `[1, 637, 4096]` | bfloat16 | 4.98 MB | 4.98 MB |
| `latents.pt` | `[1, 1024, 64]` | bfloat16 | 0.125 MB | 0.127 MB |
| `image.png` (RGBA) | 512×512×4 | uint8 | 2.00 MB (bf16 tensor) | 0.307 MB |
| `image_rgb.png` | 512×512×3 | uint8 | — | 0.269 MB |

The hand-off cost is trivial: **5.1 MB total** crosses the process boundary, against
31.6 GB of weights. The 504-word prompt tokenised to 637 positions after the
system-prompt prefix is dropped. The latent is 1024 tokens (a 32×32 grid at the VAE's 16×
spatial compression) × 64 channels — 128 KB that expands to a 512×512 image, a 2.5×
*expansion* into PNG rather than a compression, since a 128 KB latent is a denser
representation than the PNG of what it decodes to.

## The RTX 3090 (24 GB)

![Swiss Army knife penguin, RTX 3090](results_3090/image_rgb.png)

*Same prompt, same seed, same 40 steps — rendered on a 24 GB card the model cannot be
loaded onto whole. Visually indistinguishable from the 5090 output; the PNGs differ
bit-for-bit, as reduced-precision kernels on two architectures always will.*

Ran first try, in **20 minutes wall for $0.07**, on a community RTX 3090 at $0.22/hr.

### It fits, and the control proves it has to

| Stage | weights | peak device used | of 24 124 MB | activation overhead |
|---|---:|---:|---:|---:|
| `vl` | 16 722 | **17 713** | 73% | 991 |
| `dit` | 13 571 | 14 545 | 60% | 974 |
| `vae` | 644 | 2 833 | 12% | 2 189 |
| `fit` (control) | 30 937 | 22 609 | — | **OOM** |

The control run is the clean result the 5090 could not give. There, all three blocks
loaded and left 618 MB free — too little to run, but it did load. Here it dies mid-load:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB.
GPU 0 has a total capacity of 23.56 GiB of which 11.06 MiB is free.
```

Thirty-two mebibytes, with eleven free. On a 24 GB card the process split is not a
tuning choice — it is the difference between running the model and not.

### Latency, against the 5090

| | RTX 3090 | RTX 5090 |
|---|---:|---:|
| Usable VRAM | 24 124 MB | 32 109 MB |
| Compute capability | sm_86 (Ampere) | sm_120 (Blackwell) |
| Container | `pytorch/pytorch:2.13.0-cuda12.6` | `runpod/pytorch:…-cu1281-torch2130` |
| Price | $0.22/hr (community) | $0.99/hr (secure) |
| Checkpoint download | 296 s @ **106.6 MB/s** | 2 190 s @ 14.4 MB/s |
| Python import × 3 | **11.0 s** | 39.2 s |
| Read safetensors × 3 | **1.7 s** | 18.5 s |
| Host → GPU × 3 | **6.0 s** | 16.8 s |
| Model compute | 13.2 s | **9.9 s** |
| **Three-stage wall** | **32.3 s** | 85.5 s |
| Overhead / compute | **1.4×** | 7.6× |
| Cost of the run | **$0.07** | $0.72 |

The 3090 finished the pipeline in **38% of the 5090's wall time while being the slower
card**, because almost everything outside the DiT loop was the host and the container
rather than the GPU: a 3.6 GB image instead of 11.3 GB, a host with 7× the download
bandwidth, and a faster disk.

### Where the 5090 actually wins

Only one number here is close to a like-for-like GPU comparison — the DiT's steady-state
step time, which is the same code on the same torch 2.13.0 doing the same work 40 times:

| DiT, 40 steps @ 512×512 | RTX 3090 | RTX 5090 |
|---|---:|---:|
| First step | 0.720 s | 1.043 s |
| Median step | **0.283 s** | **0.107 s** |
| First ÷ median | 2.5× | 9.7× |
| Total denoise | 11.75 s | 5.85 s |

The 5090 is **2.64× faster per step**, which is the ordering the hardware predicts. But
look at the first step: the 5090 spends **0.936 s** above its median on step one against
the 3090's 0.437 s — the faster card pays twice the absolute warm-up. Part of that is
algorithmic and identical on both (the text KV-cache prefill that `causal_condition`
makes possible), so the extra is most likely PTX JIT: the cu129 build almost certainly
ships no sm_120 cubins, so every kernel compiles on first launch.

That also explains the two stages where the slower card beat the faster one — VL encode
1.21 s against 2.80 s, VAE decode 0.28 s against 1.25 s. Both are **one-shot**: they
launch each kernel once and pay warm-up in full, with no loop to amortise it over. The
DiT runs 40 iterations and buries it.

Caveat: the two runs differ in GPU, container image and host at once, so only the
steady-state step time above should be read as a GPU-to-GPU result. The rest is a
statement about how much of a short pipeline run is not the GPU.

## How it works

`run_stage.py` runs one stage per invocation. The split relies on diffusers skipping any
component passed explicitly as `None`, so each process reads only the shards it needs:

```python
# vl:  text encoder only        dit: transformer only         vae: decoder only
QwenImage21Pipeline.from_pretrained(   QwenImage21Pipeline.from_pretrained(
    MODEL_DIR, transformer=None,           MODEL_DIR, text_encoder=None,
    vae=None, dtype=torch.bfloat16)        vae=None, dtype=torch.bfloat16)
```

Stage 1 calls `pipe.encode_prompt()` and saves the hidden states. Stage 2 feeds those
back in as `prompt_embeds=` with `output_type="latent"`, so the pipeline never touches
the VAE. Stage 3 loads `AutoencoderKLQwenImage21` alone and reproduces the pipeline's
unpack → denormalise → decode tail by hand.

The processor (15 MB of tokenizer) is loaded in all three, because
`QwenImage21Pipeline.__init__` derives its prompt-template token offsets from it.

Files:

- `prompt.txt` — the pre-expanded prompt
- `run_stage.py` — the stages and all instrumentation
- `run_all.sh` — pod-side orchestrator, one process per stage
- `boot.sh` — container entrypoint
- `deploy.py` — RunPod lifecycle (create / status / pull / kill)
- `results/` — image, logs, per-stage metrics JSON

## What went wrong

### RTX 5090 — five pods, by hand

1. **Multi-line container start command** (`$0.36`, 32 min). RunPod stores `dockerArgs`
   verbatim, and a value containing newlines leaves the container unable to start — no
   error, just `runtime: null` forever. Fixed by reducing `dockerArgs` to a single line
   that clones the repo and hands off to `boot.sh`.
2. **CUDA forward-compatibility trap** (`$0.08`). The `cu1290` image on a host with a 570
   driver makes torch link the CUDA forward-compat libraries, which GeForce cards do not
   support: `cudaGetDeviceCount` fails with *error 804*, `torch.cuda.is_available()`
   returns `False`, and everything silently falls back to CPU. Fixed by pinning the
   `cu1281` image, and the `env` stage now aborts before downloading 33 GB if the GPU
   isn't visible.
3. **`threading.Thread._stop` collision** (`$0.08`). The memory samplers named their
   shutdown `Event` `self._stop`, shadowing a private method `Thread.join` calls. Every
   stage computed correctly and then crashed in teardown with
   *"'Event' object is not callable"*.
4. **A host that never finished pulling the image** (`$0.25`, 15 min). Recreated; landed
   on the same machine, which then started in under a minute from its warm layer cache.
5. **Success** (`$0.72`, 43 min — 36 of them downloading weights).

Subtotal: **$1.49**.

### RTX 3090 — one command, nine pods, mostly automatic

The 3090 run used the hardened `deploy.py run`, which rents, watches, collects and
terminates on its own. Nine pods went by; only the first two failure *classes* needed me.

| | Pods | Cost | Handled by |
|---|---:|---:|---|
| Secure host (before switching to community) | 1 | $0.02 | — |
| Boot timeout, 11.3 GB image never pulled | 2 | $0.08 | `BOOT_TIMEOUT` + rotation |
| Landed on a known-bad machine | 3 | $0.01 | machine blacklist, released in seconds |
| GPU passthrough broken (`CUDA unknown error`) | 1 | $0.01 | CUDA preflight → `FAILED:cuda` |
| PEP 668 blocked every `pip install` | 1 | $0.01 | code fix, then fatal + verified |
| **Success** | 1 | **$0.07** | — |

Two of those became permanent fixes rather than incidents: the 3.6 GB base image (which
also removed the error-804 trap outright) and the persistent bad-machine list. The
`SUPPLY_CONSTRAINT` wait and the PEP 668 flag were the remaining code fixes.

Subtotal: **$0.20**. Grand total across both cards: **$1.69**.

## Takeaways

- **On 24 GB the split is mandatory.** The three blocks are 30.9 GB in bf16; loading
  them together on a 3090 dies trying to allocate 32 MiB with 11 MiB free. Per-stage
  peak is 17.7 GB, 73% of the card.
- **On 32 GB it is still necessary, more insidiously.** They load, leaving 618 MB — less
  than any stage's activation working set (991 MB / 974 MB / 2 189 MB), so a monolithic
  run OOMs on the first forward pass instead of at load.
- **The split is nearly free in bytes moved**: 5.1 MB of intermediate tensors against
  31.6 GB of weights.
- **Most of a short run is not the GPU.** On the 5090 the fixed cost was 75.6 s against
  9.9 s of compute (7.6×); on the 3090, with a 3× smaller image and a faster host, the
  same fixed cost was 19.0 s against 13.2 s (1.4×). Same code, same split — the
  difference is the container and the machine.
- **A slower card can finish sooner.** The 3090 is 2.64× slower per DiT step and still
  completed the pipeline in 38% of the 5090's wall time, at a tenth of the cost.
- **One-shot kernels punish new architectures.** The 5090 lost VL encode (2.80 s vs
  1.21 s) and VAE decode (1.25 s vs 0.28 s) to the 3090, and spent 0.936 s above median
  on its first DiT step against the 3090's 0.437 s — consistent with the cu129 build
  shipping no sm_120 cubins and PTX-JITing each kernel on first launch. Only the 40-step
  DiT loop runs long enough to amortise it.
- **Scale changes the arithmetic.** 2048×2048 is 16× the latent tokens, so DiT compute
  goes from ~6–12 s to minutes — more, since attention over image tokens is quadratic —
  while the overhead stays flat. The split gets cheaper the closer you run to the
  resolution the model was built for.
- Use this for batch work, where the fixed cost amortises across many prompts per stage.
  For one-off interactive generation, `enable_model_cpu_offload()` is the simpler trade
  — but only on a card where all three blocks fit in host RAM *and* one fits in VRAM.
