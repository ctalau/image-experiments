#!/usr/bin/env python3
"""Drive the experiment on a RunPod GPU pod.

  deploy.py create     -- rent a 32 GB pod and start the run
  deploy.py status     -- one line of state + the tail of the run log
  deploy.py log        -- the whole run log
  deploy.py pull DIR   -- download artifacts and metrics
  deploy.py kill       -- terminate the pod

The pod exposes a read-only static file server on port 8000 through RunPod's
HTTPS proxy; that is the only channel back, which keeps the control plane to
"download a file" rather than "run a command".
"""

import json
import os
import sys
import time
import urllib.request

API = "https://api.runpod.io/graphql"
KEY = os.environ["RUNPOD_KEY"]
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pod.json")

GPU_TYPE = os.environ.get("GPU_TYPE", "NVIDIA GeForce RTX 5090")
# CUDA 12.8, not 12.9: RunPod's RTX 5090 hosts run a 570 driver, and a cu129
# build only reaches it through CUDA forward compatibility, which GeForce
# cards do not support (`cudaGetDeviceCount` fails with error 804).
IMAGE = os.environ.get(
    "POD_IMAGE", "runpod/pytorch:1.3.2-cu1281-torch2130-ubuntu2404"
)
REPO = os.environ.get("REPO_URL", "https://github.com/ctalau/image-experiments")
BRANCH = os.environ.get("REPO_BRANCH", "claude/qwen-image-runpod-sequential-xr5skn")

# The container start command has to stay on one line with no shell quoting:
# RunPod stores it verbatim and a multi-line value stops the container from
# starting at all. Everything else lives in boot.sh inside the repo.
BOOT = (
    "bash -c "
    "'apt-get update -qq; apt-get install -y -qq git; "
    f"git clone --depth 1 -b {BRANCH} {REPO} /workspace/repo; "
    "bash /workspace/repo/qwen_image_21_sequential/boot.sh'"
)


def gql(query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
            # Cloudflare in front of the API rejects the default urllib UA (1010).
            "User-Agent": "curl/8.5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    if "errors" in out:
        raise SystemExit(json.dumps(out["errors"], indent=2))
    return out["data"]


def save_state(d):
    with open(STATE, "w") as f:
        json.dump(d, f, indent=2)


def load_state():
    with open(STATE) as f:
        return json.load(f)


def create():
    docker_args = BOOT
    data = gql(
        """
        mutation ($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id name imageName machineId costPerHr
            machine { podHostId gpuDisplayName }
          }
        }
        """,
        {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "gpuTypeId": GPU_TYPE,
                "name": "qwen-image-21-sequential",
                "imageName": IMAGE,
                "containerDiskInGb": 80,
                "volumeInGb": 0,
                "minVcpuCount": 8,
                "minMemoryInGb": 32,
                "ports": "8000/http",
                "dockerArgs": docker_args,
                "env": [],
            }
        },
    )
    pod = data["podFindAndDeployOnDemand"]
    pod["started_at"] = time.time()
    pod["base_url"] = f"https://{pod['id']}-8000.proxy.runpod.net"
    save_state(pod)
    print(json.dumps(pod, indent=2))


def pod_info(pod_id):
    return gql(
        """
        query ($id: String!) {
          pod(input: {podId: $id}) {
            id desiredStatus costPerHr lastStatusChange
            runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } }
          }
        }
        """,
        {"id": pod_id},
    )["pod"]


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}: {e}>".encode()


def status():
    st = load_state()
    info = pod_info(st["id"])
    base = st["base_url"]
    print(json.dumps(info, indent=2))
    print(f"elapsed: {time.time() - st['started_at']:.0f}s   cost so far: "
          f"${(time.time() - st['started_at']) / 3600 * (info.get('costPerHr') or 0):.3f}")
    print("STATUS:", fetch(base + "/STATUS", 30).decode(errors="replace").strip())
    log = fetch(base + "/run.log", 60).decode(errors="replace")
    print("--- last 60 log lines ---")
    print("\n".join(log.splitlines()[-60:]))


def log():
    st = load_state()
    sys.stdout.write(fetch(st["base_url"] + "/run.log", 120).decode(errors="replace"))


def pull(dest):
    st = load_state()
    base = st["base_url"]
    os.makedirs(dest, exist_ok=True)
    names = ["run.log", "metrics.json", "STATUS",
             "artifacts/image.png", "artifacts/image_rgb.png"]
    for n in names:
        data = fetch(f"{base}/{n}", 180)
        if data.startswith(b"<"):
            print(f"skip {n}: {data[:120].decode(errors='replace')}")
            continue
        p = os.path.join(dest, os.path.basename(n))
        with open(p, "wb") as f:
            f.write(data)
        print(f"saved {p} ({len(data)} bytes)")


def kill():
    st = load_state()
    print(gql("mutation ($id: String!) { podTerminate(input: {podId: $id}) }",
              {"id": st["id"]}))
    st["terminated_at"] = time.time()
    st["billed_seconds"] = round(st["terminated_at"] - st["started_at"])
    st["billed_usd"] = round(st["billed_seconds"] / 3600 * st.get("costPerHr", 0), 4)
    save_state(st)
    print(json.dumps({k: st[k] for k in ("id", "billed_seconds", "billed_usd")}, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "pull":
        pull(sys.argv[2])
    elif cmd in ("create", "status", "log", "kill"):
        {"create": create, "status": status, "log": log, "kill": kill}[cmd]()
    else:
        raise SystemExit(__doc__)
