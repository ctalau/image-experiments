#!/usr/bin/env python3
"""Turn the collected metrics into the tables used in REPORT.md."""

import json
import sys


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main(path):
    m = json.load(open(path))
    out = []

    env = m.get("env", {})
    out.append("### Environment\n")
    out.append(row("Field", "Value"))
    out.append(row("---", "---"))
    for k in ("gpu", "gpu_total_gb", "capability", "host_ram_gb", "cpu_count",
              "torch", "cuda", "diffusers", "transformers", "accelerate"):
        if k in env:
            out.append(row(k, env[k]))

    out.append("\n### Stage latency (s)\n")
    out.append(row("Stage", "load weights (CPU)", "host -> GPU", "compute", "save", "total"))
    out.append(row("---", "---", "---", "---", "---", "---"))
    compute_key = {"vl": "encode", "dit": "denoise", "vae": "decode"}
    for st in ("vl", "dit", "vae"):
        d = m.get(st, {}).get("latency_s", {})
        out.append(row(st, d.get("load_cpu", "-"), d.get("to_gpu", "-"),
                       d.get(compute_key[st], "-"), d.get("save", "-"), d.get("total", "-")))

    out.append("\n### Memory (MB)\n")
    out.append(row("Stage", "weights", "torch peak alloc", "torch peak reserved",
                   "peak device used", "peak host RSS"))
    out.append(row("---", "---", "---", "---", "---", "---"))
    wkey = {"vl": "text_encoder", "dit": "transformer", "vae": "vae"}
    for st in ("vl", "dit", "vae"):
        d = m.get(st, {})
        g, h = d.get("gpu", {}), d.get("host", {})
        out.append(row(st, d.get(wkey[st], {}).get("param_bytes_mb", "-"),
                       g.get("torch_peak_allocated_mb", "-"),
                       g.get("torch_peak_reserved_mb", "-"),
                       g.get("peak_device_used_mb", "-"),
                       h.get("peak_rss_mb", "-")))

    out.append("\n### Intermediate artifacts\n")
    out.append(row("Artifact", "Shape", "dtype", "In-memory MB", "On-disk MB"))
    out.append(row("---", "---", "---", "---", "---"))
    vl = m.get("vl", {})
    pe = vl.get("outputs", {}).get("prompt_embeds", {})
    out.append(row("prompt_embeds.pt", pe.get("shape"), pe.get("dtype"),
                   pe.get("bytes_mb"), vl.get("artifact", {}).get("size_mb")))
    dit = m.get("dit", {})
    la = dit.get("outputs", {}).get("latents", {})
    out.append(row("latents.pt", la.get("shape"), la.get("dtype"),
                   la.get("bytes_mb"), dit.get("artifact", {}).get("size_mb")))
    vae = m.get("vae", {})
    o = vae.get("outputs", {})
    out.append(row("image.png", o.get("image_size"), o.get("image_mode"), "-",
                   vae.get("artifact", {}).get("size_mb")))

    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/metrics.json")
