# Qwen-Image-2.1 on a single 32 GB GPU, run stage by stage

Running [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) text-to-image on one
RunPod RTX 5090 (32 GB), with the three weight blocks — the Qwen3-VL text encoder, the
DiT, and the VAE — executed as **three separate processes** that hand each other tensors
on disk, so only one block is on the GPU at a time.

Target: a 512×512 image from a pre-expanded prompt for *"swiss army knife penguin"*.

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

Usable VRAM on the card is 32 109 MB (nvidia-smi reports 32 607 MiB installed). So the
weights do fit, with 1 172 MB to spare — and that is the whole problem, which the control
run below makes concrete.

## Results

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

Five pods were rented before one produced the image.

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

Total GPU spend: **$1.49**. The successful run's useful work was 85 s.

## Takeaways

- Qwen-Image-2.1's three blocks total 30.9 GB in bf16 and *technically* fit on a 32 GB
  card, but with only 618 MB of headroom — not enough to run. A stage-per-process split
  brings peak VRAM down to 17.6 GB, 55% of the card, leaving room for 2K generation.
- The split is nearly free in bytes moved: 5.1 MB of intermediate tensors versus 31.6 GB
  of weights.
- It is not free in time. At 512×512 the fixed cost (imports, safetensors reads,
  host→device copies) is 75 s against 9.9 s of compute — a 7.6× overhead. That ratio
  collapses at the resolution the model is built for: 2048×2048 is 16× the latent tokens,
  so DiT compute alone would go from ~6 s to at least a minute and a half — more, since
  attention over image tokens is quadratic — while the overhead stays flat.
- Use this for batch work, where the fixed cost amortises across many prompts per stage.
  For one-off interactive generation on a too-small card, `enable_model_cpu_offload()`
  is the better trade.
