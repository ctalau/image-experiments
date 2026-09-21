#!/usr/bin/env python3
"""Run one stage of Qwen-Image-2.1 text-to-image in its own process.

Qwen-Image-2.1 ships three weight blocks that together exceed a 32 GB card:
a Qwen3-VL text encoder (~17.5 GB bf16), a 32-layer single-stream DiT
(~14.2 GB bf16) and a 64-channel VAE (~1.35 GB bf16).  Running them in one
process means all three are resident at once.  Here each stage runs as its
own process and hands the next one a tensor on disk, so only one block is
ever on the GPU.

  fetch  -> download the checkpoint
  vl     -> Qwen3-VL encodes the prompt        -> prompt_embeds.pt
  dit    -> DiT denoises from the embeddings   -> latents.pt
  vae    -> VAE decodes the latents            -> image.png
  fit    -> control: try to hold all three on the GPU at once

Every stage writes a JSON metrics file next to its artifact.
"""

import argparse
import json
import os
import resource
import subprocess
import threading
import time

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/models/Qwen-Image-2.1")
ART_DIR = os.environ.get("ART_DIR", "/workspace/out/artifacts")
METRICS_DIR = os.environ.get("METRICS_DIR", "/workspace/out/metrics")

HEIGHT = int(os.environ.get("HEIGHT", "512"))
WIDTH = int(os.environ.get("WIDTH", "512"))
STEPS = int(os.environ.get("STEPS", "40"))
SEED = int(os.environ.get("SEED", "42"))
PROMPT_FILE = os.environ.get("PROMPT_FILE", "prompt.txt")

MB = 1024.0 * 1024.0


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """Poll whole-device memory use, which includes the CUDA context and any
    allocation torch does not see. `torch.cuda.max_memory_allocated` only
    covers the caching allocator, so it understates what the card needs."""

    daemon = True

    def __init__(self, period=0.2):
        super().__init__()
        self.period = period
        self.peak_used = 0
        self.total = 0
        self._stop = threading.Event()

    def run(self):
        import torch

        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            self.total = total
            self.peak_used = max(self.peak_used, total - free)
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


class HostSampler(threading.Thread):
    """Peak RSS of this process, sampled from /proc (ru_maxrss only updates
    on some events and misses short-lived peaks on some kernels)."""

    daemon = True

    def __init__(self, period=0.2):
        super().__init__()
        self.period = period
        self.peak_rss = 0
        self._stop = threading.Event()

    def _rss(self):
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return 0

    def run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self._rss())
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks = {}
        self._last = self.t0

    def mark(self, name):
        now = time.perf_counter()
        self.marks[name] = round(now - self._last, 3)
        self._last = now
        print(f"  [{name}] {self.marks[name]:.3f}s", flush=True)

    @property
    def total(self):
        return round(time.perf_counter() - self.t0, 3)


def host_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024.0 / 1024.0, 2)
    except Exception:
        pass
    return None


def module_stats(module):
    params = sum(p.numel() for p in module.parameters())
    pbytes = sum(p.numel() * p.element_size() for p in module.parameters())
    bbytes = sum(b.numel() * b.element_size() for b in module.buffers())
    dtypes = sorted({str(p.dtype) for p in module.parameters()})
    return {
        "class": type(module).__name__,
        "params": params,
        "params_billions": round(params / 1e9, 3),
        "param_bytes_mb": round(pbytes / MB, 1),
        "buffer_bytes_mb": round(bbytes / MB, 1),
        "dtypes": dtypes,
    }


def tensor_stats(t):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "elements": int(t.numel()),
        "bytes_mb": round(t.numel() * t.element_size() / MB, 3),
    }


def write_metrics(stage, payload):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, f"{stage}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n=== metrics/{stage}.json ===")
    print(json.dumps(payload, indent=2), flush=True)


def gpu_name():
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def finish(stage, timer, gpu, host, extra):
    import torch

    gpu.stop()
    host.stop()
    payload = {
        "stage": stage,
        "latency_s": dict(timer.marks, total=timer.total),
        "gpu": {
            "name": gpu_name(),
            "total_mb": round(gpu.total / MB, 1),
            "peak_device_used_mb": round(gpu.peak_used / MB, 1),
            "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / MB, 1),
            "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / MB, 1),
        },
        "host": {
            "peak_rss_mb": round(host.peak_rss / MB, 1),
            "ru_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
            "total_ram_gb": host_ram_gb(),
        },
    }
    payload.update(extra)
    write_metrics(stage, payload)


def start_samplers():
    import torch  # noqa: F401  (GpuSampler needs cuda initialised by caller)

    gpu = GpuSampler()
    host = HostSampler()
    host.start()
    gpu.start()
    return gpu, host


def dtype_bf16():
    import torch

    return torch.bfloat16


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_fetch():
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    t = Timer()
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["assets/*", "*.md"],
        max_workers=8,
    )
    t.mark("download")

    sizes = {}
    for sub in ("text_encoder", "transformer", "vae", "processor", "scheduler"):
        p = os.path.join(MODEL_DIR, sub)
        total = 0
        for root, _, files in os.walk(p):
            for fn in files:
                fp = os.path.join(root, fn)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        sizes[sub] = round(total / MB, 1)
    sizes["TOTAL"] = round(sum(sizes.values()), 1)

    write_metrics(
        "fetch",
        {
            "stage": "fetch",
            "model_id": MODEL_ID,
            "latency_s": dict(t.marks, total=t.total),
            "checkpoint_size_mb": sizes,
            "download_mb_per_s": round(sizes["TOTAL"] / max(t.marks["download"], 1e-6), 1),
        },
    )


def stage_vl():
    """Qwen3-VL encodes the prompt into per-token hidden states."""
    t = Timer()
    import torch
    from diffusers import QwenImage21Pipeline

    t.mark("import")
    torch.cuda.init()
    gpu, host = start_samplers()

    with open(PROMPT_FILE) as f:
        prompt = f.read().strip()

    # transformer and vae stay unloaded: `from_pretrained` skips any component
    # explicitly passed as None, so their shards are never read.
    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL_DIR, transformer=None, vae=None, dtype=dtype_bf16()
    )
    t.mark("load_cpu")

    pipe.text_encoder.to("cuda")
    torch.cuda.synchronize()
    t.mark("to_gpu")

    enc_stats = module_stats(pipe.text_encoder)

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
            prompt=prompt, device=torch.device("cuda")
        )
    torch.cuda.synchronize()
    t.mark("encode")

    payload = {
        "prompt_chars": len(prompt),
        "prompt_words": len(prompt.split()),
        "text_encoder": enc_stats,
        "outputs": {
            "prompt_embeds": tensor_stats(prompt_embeds),
            "prompt_embeds_mask": None if prompt_embeds_mask is None else tensor_stats(prompt_embeds_mask),
            "image_pad_mask": tensor_stats(image_pad_mask),
        },
    }

    os.makedirs(ART_DIR, exist_ok=True)
    out = os.path.join(ART_DIR, "prompt_embeds.pt")
    torch.save(
        {
            "prompt_embeds": prompt_embeds.cpu(),
            "prompt_embeds_mask": None if prompt_embeds_mask is None else prompt_embeds_mask.cpu(),
            "image_pad_mask": image_pad_mask.cpu(),
        },
        out,
    )
    t.mark("save")
    payload["artifact"] = {"path": out, "size_mb": round(os.path.getsize(out) / MB, 3)}

    finish("vl", t, gpu, host, payload)


def stage_dit():
    """The DiT denoises 512x512 latents from the text embeddings alone."""
    t = Timer()
    import torch
    from diffusers import QwenImage21Pipeline

    t.mark("import")
    torch.cuda.init()
    gpu, host = start_samplers()

    blob = torch.load(os.path.join(ART_DIR, "prompt_embeds.pt"), map_location="cpu")
    prompt_embeds = blob["prompt_embeds"].to("cuda")
    mask = blob["prompt_embeds_mask"]
    if mask is not None:
        mask = mask.to("cuda")
    t.mark("load_embeds")

    # text_encoder and vae stay unloaded; the processor is still needed because
    # the pipeline derives its prompt-template offsets from the tokenizer.
    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL_DIR, text_encoder=None, vae=None, dtype=dtype_bf16()
    )
    t.mark("load_cpu")

    pipe.transformer.to("cuda")
    torch.cuda.synchronize()
    t.mark("to_gpu")

    dit_stats = module_stats(pipe.transformer)

    step_times = []

    def on_step(p, i, tstep, kw):
        torch.cuda.synchronize()
        step_times.append(time.perf_counter())
        return kw

    gen = torch.Generator(device="cuda").manual_seed(SEED)
    t_denoise0 = time.perf_counter()
    latents = pipe(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=mask,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        generator=gen,
        output_type="latent",
        callback_on_step_end=on_step,
    ).images
    torch.cuda.synchronize()
    t.mark("denoise")

    marks = [t_denoise0] + step_times
    per_step = [round(marks[i + 1] - marks[i], 4) for i in range(len(marks) - 1)]

    os.makedirs(ART_DIR, exist_ok=True)
    out = os.path.join(ART_DIR, "latents.pt")
    torch.save(latents.cpu(), out)
    t.mark("save")

    payload = {
        "resolution": {"height": HEIGHT, "width": WIDTH},
        "num_inference_steps": STEPS,
        "seed": SEED,
        "transformer": dit_stats,
        "inputs": {"prompt_embeds": tensor_stats(prompt_embeds)},
        "outputs": {"latents": tensor_stats(latents)},
        "per_step_s": {
            "first": per_step[0] if per_step else None,
            "median": round(sorted(per_step)[len(per_step) // 2], 4) if per_step else None,
            "mean": round(sum(per_step) / len(per_step), 4) if per_step else None,
            "all": per_step,
        },
        "artifact": {"path": out, "size_mb": round(os.path.getsize(out) / MB, 3)},
    }
    finish("dit", t, gpu, host, payload)


def stage_vae():
    """The VAE decoder turns the 64-channel latents into pixels."""
    t = Timer()
    import torch
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    t.mark("import")
    torch.cuda.init()
    gpu, host = start_samplers()

    latents = torch.load(os.path.join(ART_DIR, "latents.pt"), map_location="cpu")
    t.mark("load_latents")

    vae = AutoencoderKLQwenImage21.from_pretrained(MODEL_DIR, subfolder="vae", dtype=dtype_bf16())
    t.mark("load_cpu")

    vae.to("cuda").eval()
    torch.cuda.synchronize()
    t.mark("to_gpu")

    vae_stats = module_stats(vae)
    scale = 16
    z_dim = vae.config.z_dim

    # Undo the pipeline's spatial flatten, then the latent normalisation the
    # checkpoint was trained with, before decoding.
    lat = latents.to("cuda", dtype=dtype_bf16())
    b, _, c = lat.shape
    h = 2 * (HEIGHT // (scale * 2))
    w = 2 * (WIDTH // (scale * 2))
    lat = lat.transpose(1, 2).reshape(b, c, 1, h, w)
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(lat.device, lat.dtype)
    lat = lat * std + mean
    unpacked = tensor_stats(lat)

    with torch.no_grad():
        decoded = vae.decode(lat, return_dict=False)[0][:, :, 0]
    torch.cuda.synchronize()
    t.mark("decode")

    proc = VaeImageProcessor(vae_scale_factor=scale, vae_latent_channels=z_dim)
    image = proc.postprocess(decoded.float(), output_type="pil")[0]

    os.makedirs(ART_DIR, exist_ok=True)
    out = os.path.join(ART_DIR, "image.png")
    image.save(out)
    # The VAE is 4-channel (the checkpoint generates RGBA natively), so also
    # keep a copy flattened onto white for embedding in documents.
    flat_path = None
    alpha_min = None
    if image.mode == "RGBA":
        alpha_min = int(min(image.getchannel("A").getdata()))
        from PIL import Image as _Image

        white = _Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.getchannel("A"))
        flat_path = os.path.join(ART_DIR, "image_rgb.png")
        white.save(flat_path)
    t.mark("save")

    payload = {
        "vae": vae_stats,
        "inputs": {"packed_latents": tensor_stats(latents), "unpacked_latents": unpacked},
        "outputs": {
            "decoded_tensor": tensor_stats(decoded),
            "image_mode": image.mode,
            "image_size": list(image.size),
            "alpha_min": alpha_min,
        },
        "artifact": {"path": out, "size_mb": round(os.path.getsize(out) / MB, 3)},
        "artifact_flat": None
        if flat_path is None
        else {"path": flat_path, "size_mb": round(os.path.getsize(flat_path) / MB, 3)},
        "compression_ratio_latent_to_png": round(
            latents.numel() * latents.element_size() / os.path.getsize(out), 3
        ),
    }
    finish("vae", t, gpu, host, payload)


def stage_fit():
    """Control: load all three blocks onto the GPU at once and see what happens."""
    import torch
    from diffusers import QwenImage21Pipeline

    torch.cuda.init()
    gpu, host = start_samplers()
    t = Timer()
    result = {"stage": "fit", "oom": False, "error": None}
    try:
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_DIR, dtype=dtype_bf16())
        t.mark("load_cpu")
        pipe.to("cuda")
        torch.cuda.synchronize()
        t.mark("to_gpu")
        result["weights_mb"] = {
            "text_encoder": module_stats(pipe.text_encoder)["param_bytes_mb"],
            "transformer": module_stats(pipe.transformer)["param_bytes_mb"],
            "vae": module_stats(pipe.vae)["param_bytes_mb"],
        }
    except torch.cuda.OutOfMemoryError as e:
        result["oom"] = True
        result["error"] = str(e)[:600]
        print(f"OOM as expected: {str(e)[:300]}", flush=True)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        print(f"failed: {result['error']}", flush=True)
    finish("fit", t, gpu, host, result)


def stage_env():
    import torch

    info = {
        "stage": "env",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name(),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if torch.cuda.is_available()
        else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "host_ram_gb": host_ram_gb(),
        "cpu_count": os.cpu_count(),
    }
    for mod in ("diffusers", "transformers", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            info[mod] = f"missing: {e}"
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        pass
    write_metrics("env", info)


STAGES = {
    "env": stage_env,
    "fetch": stage_fetch,
    "vl": stage_vl,
    "dit": stage_dit,
    "vae": stage_vae,
    "fit": stage_fit,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    args = ap.parse_args()
    print(f"\n########## stage: {args.stage} ##########", flush=True)
    STAGES[args.stage]()
