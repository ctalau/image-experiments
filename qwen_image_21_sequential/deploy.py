#!/usr/bin/env python3
"""Rent a RunPod GPU, run the sequential Qwen-Image-2.1 pipeline, collect the
results, and terminate the pod.

  deploy.py run [--gpu NAME] [--cloud C] [--out DIR]   end to end (use this)
  deploy.py create [--gpu NAME]            rent a pod and start the run
  deploy.py status                         pod state + tail of the run log
  deploy.py log                            the whole run log
  deploy.py pull DIR                       download artifacts and metrics
  deploy.py kill                           terminate the pod from .pod.json
  deploy.py reap                           terminate EVERY pod on the account

`run` is the supported entry point. It rotates to a different host on the two
failures that are the host's fault rather than the code's -- a container that
never starts, and a GPU torch cannot open -- and it terminates the pod on every
exit path including Ctrl-C, so a failed run cannot leave a GPU billing.

The pod exposes a read-only static file server on port 8000 through RunPod's
HTTPS proxy; that is the only channel back, which keeps the control plane at
"download a file" rather than "run a command".
"""

import argparse
import atexit
import json
import os
import signal
import sys
import time
import urllib.request

API = "https://api.runpod.io/graphql"
KEY = os.environ["RUNPOD_KEY"]
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".pod.json")
# Machines that failed at host level, remembered across runs so a later
# invocation does not rediscover them the slow way.
BADLIST = os.path.join(HERE, ".bad-machines.json")

POD_NAME = "qwen-image-21-sequential"
DEFAULT_GPU = "NVIDIA GeForce RTX 3090"
REPO = os.environ.get("REPO_URL", "https://github.com/ctalau/image-experiments")
BRANCH = os.environ.get("REPO_BRANCH", "claude/qwen-image-runpod-sequential-xr5skn")

# Upstream pytorch/pytorch rather than runpod/pytorch: 3.6 GB compressed
# against 11.3 GB, and community hosts routinely fail to pull the larger one
# inside BOOT_TIMEOUT. It also carries no CUDA forward-compatibility libraries,
# which removes the error-804 trap outright (boot.sh still guards for it).
# CUDA 12.6 is the newest cu12 build published for torch 2.13 and runs on any
# driver from 525 up through CUDA minor-version compatibility.
IMAGE = os.environ.get("POD_IMAGE", "pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime")

# RunPod stores the container start command verbatim, and a value containing
# newlines leaves the container unable to start with no error anywhere. Keep
# this to one line; everything else lives in boot.sh inside the repo.
BOOT = (
    "bash -c "
    "'apt-get update -qq; apt-get install -y -qq git; "
    f"git clone --depth 1 -b {BRANCH} {REPO} /workspace/repo; "
    "bash /workspace/repo/qwen_image_21_sequential/boot.sh'"
)

# Statuses the pod writes to /workspace/out/STATUS.
# Host-level failures worth retrying elsewhere. Everything else (a failed
# install, a stage that raised) is the code's fault and rotating just burns
# money re-running the same bug.
ROTATE = ("FAILED:cuda",)
ARTIFACTS = ["run.log", "metrics.json", "STATUS",
             "artifacts/image.png", "artifacts/image_rgb.png"]

BOOT_TIMEOUT = 660      # no STATUS file at all -> the image never finished pulling
RUN_TIMEOUT = 7200      # DONE never arrives
POLL = 15
CAPACITY_WAIT = 1800    # how long to wait out "no instances available"
CAPACITY_POLL = 45
BADLIST_TTL = 86400     # a host bad today may be fine tomorrow

_live = set()           # pod ids this process created and has not terminated


# --------------------------------------------------------------------------
# RunPod API
# --------------------------------------------------------------------------
class RunPodError(Exception):
    def __init__(self, errors):
        super().__init__(json.dumps(errors, indent=2))
        self.errors = errors

    @property
    def codes(self):
        return {(e.get("extensions") or {}).get("code") for e in self.errors}


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
        raise RunPodError(out["errors"])
    return out["data"]


def deploy(gpu_type, cloud="COMMUNITY", capacity_wait=CAPACITY_WAIT):
    """Rent one pod, waiting out transient 'no instances available'.

    Community stock for a given card comes and goes minute to minute, so a
    SUPPLY_CONSTRAINT is something to sit through, not to fail the run on."""
    deadline = time.time() + capacity_wait
    while True:
        try:
            return _deploy_once(gpu_type, cloud)
        except RunPodError as e:
            if "SUPPLY_CONSTRAINT" not in e.codes or time.time() > deadline:
                raise
            left = deadline - time.time()
            print(f"  no {cloud.lower()} {gpu_type} free; retrying for {left / 60:.0f} more min",
                  flush=True)
            time.sleep(CAPACITY_POLL)


def _deploy_once(gpu_type, cloud):
    pod = gql(
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
                "cloudType": cloud,
                "gpuCount": 1,
                "gpuTypeId": gpu_type,
                "name": POD_NAME,
                "imageName": IMAGE,
                "containerDiskInGb": 80,
                "volumeInGb": 0,
                "minVcpuCount": 8,
                "minMemoryInGb": 32,
                "ports": "8000/http",
                "dockerArgs": BOOT,
                "env": [],
            }
        },
    )["podFindAndDeployOnDemand"]
    if not pod:
        raise RunPodError([{"message": f"no {cloud} capacity for {gpu_type!r}",
                            "extensions": {"code": "SUPPLY_CONSTRAINT"}}])
    pod["gpu_type"] = gpu_type
    pod["cloud"] = cloud
    pod["started_at"] = time.time()
    pod["base_url"] = f"https://{pod['id']}-8000.proxy.runpod.net"
    _live.add(pod["id"])
    save_state(pod)
    return pod


def terminate(pod_id):
    try:
        gql("mutation ($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})
    except Exception as e:  # noqa: BLE001
        print(f"!! could not terminate {pod_id}: {e}", file=sys.stderr)
        return False
    _live.discard(pod_id)
    print(f"terminated {pod_id}")
    return True


def runtime_of(pod_id):
    return gql(
        "query ($id: String!) { pod(input: {podId: $id}) "
        "{ desiredStatus costPerHr runtime { uptimeInSeconds } } }",
        {"id": pod_id},
    )["pod"]


def _reap_live(*_):
    """Terminate anything this process still owns, on any exit path."""
    for pod_id in list(_live):
        terminate(pod_id)


atexit.register(_reap_live)
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, lambda s, f: sys.exit(130))


# --------------------------------------------------------------------------
# pod file server
# --------------------------------------------------------------------------
def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}: {e}>".encode()


def status_of(base_url):
    data = fetch(base_url + "/STATUS", 25)
    if data.startswith(b"<"):
        return None                       # server not up yet
    return data.decode(errors="replace").strip()


def load_badlist():
    try:
        with open(BADLIST) as f:
            return {k: v for k, v in json.load(f).items()
                    if time.time() - v.get("at", 0) < BADLIST_TTL}
    except Exception:  # noqa: BLE001
        return {}


def record_bad(machine, reason):
    bad = load_badlist()
    bad[machine] = {"reason": reason, "at": time.time()}
    try:
        with open(BADLIST, "w") as f:
            json.dump(bad, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not persist badlist: {e})")


def save_state(d):
    with open(STATE, "w") as f:
        json.dump(d, f, indent=2)


def load_state():
    with open(STATE) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_run(args):
    # Community hosts fail in ways that are entirely the host's: a GPU the
    # driver cannot open, an image that never finishes pulling. RunPod's
    # scheduler is sticky and will hand back the same machine, so remember the
    # bad ones and drop straight back out of them -- that costs seconds and a
    # cent, against the ten minutes a boot timeout takes to notice.
    bad_machines = set(load_badlist())
    if bad_machines:
        print(f"skipping {len(bad_machines)} machine(s) known bad from earlier runs")
    attempt, deploys = 0, 0
    while attempt < args.attempts and deploys < args.attempts * 4:
        pod = deploy(args.gpu, args.cloud)
        deploys += 1
        machine = pod["machineId"]
        if machine in bad_machines:
            print(f"  landed back on known-bad machine {machine}; releasing it")
            terminate(pod["id"])
            time.sleep(20)
            continue

        attempt += 1
        pod["out"] = args.out
        print(f"\n=== attempt {attempt}/{args.attempts}: {args.gpu} "
              f"on the {args.cloud.lower()} cloud ===", flush=True)
        print(f"pod {pod['id']} on {pod['machine']['gpuDisplayName']} "
              f"(machine {machine}) at ${pod['costPerHr']}/hr")
        print(f"log: {pod['base_url']}/run.log")
        try:
            outcome = watch(pod)
        finally:
            elapsed = time.time() - pod["started_at"]
            terminate(pod["id"])
            print(f"billed ~{elapsed:.0f}s ~= ${elapsed / 3600 * pod['costPerHr']:.2f}")

        if outcome == "DONE":
            print(f"\nrun finished; artifacts in {args.out}")
            return 0
        if outcome in ROTATE or outcome == "NO_BOOT":
            bad_machines.add(machine)
            record_bad(machine, outcome)
            print(f"host-level failure ({outcome}) on {machine}; "
                  f"blacklisted, rotating to another machine")
            continue
        print(f"\nrun failed: {outcome}  (see {args.out}/run.log)")
        return 1
    print(f"\ngave up after {attempt} attempts on {len(bad_machines)} bad machines")
    return 1


def watch(pod):
    """Follow one pod to a terminal state, pulling results if it succeeds."""
    base, t0, last = pod["base_url"], time.time(), None
    while True:
        elapsed = time.time() - t0
        st = status_of(base)

        if st is None:
            if elapsed > BOOT_TIMEOUT:
                print(f"no container after {elapsed:.0f}s")
                return "NO_BOOT"
        elif st != last:
            print(f"[{elapsed:6.0f}s] {st}", flush=True)
            last = st

        if st == "DONE" or (st or "").startswith("FAILED"):
            pull(pod["base_url"], os.path.join(HERE, pod.get("out", "results")))
            return st
        if elapsed > RUN_TIMEOUT:
            pull(pod["base_url"], os.path.join(HERE, pod.get("out", "results")))
            return "TIMEOUT"
        time.sleep(POLL)


def pull(base, dest):
    os.makedirs(dest, exist_ok=True)
    for name in ARTIFACTS:
        data = fetch(f"{base}/{name}", 300)
        if data.startswith(b"<"):
            print(f"  skip {name}: {data[:100].decode(errors='replace')}")
            continue
        path = os.path.join(dest, os.path.basename(name))
        with open(path, "wb") as f:
            f.write(data)
        print(f"  saved {path} ({len(data)} bytes)")


def cmd_create(args):
    pod = deploy(args.gpu, args.cloud)
    _live.discard(pod["id"])             # `create` deliberately leaves it running
    print(json.dumps(pod, indent=2))
    return 0


def cmd_status(_args):
    st = load_state()
    info = runtime_of(st["id"])
    print(json.dumps(info, indent=2))
    elapsed = time.time() - st["started_at"]
    print(f"elapsed {elapsed:.0f}s  ~= ${elapsed / 3600 * (info.get('costPerHr') or 0):.3f}")
    print("STATUS:", status_of(st["base_url"]))
    log = fetch(st["base_url"] + "/run.log", 60).decode(errors="replace")
    print("--- last 40 log lines ---")
    print("\n".join(log.splitlines()[-40:]))
    return 0


def cmd_log(_args):
    sys.stdout.write(fetch(load_state()["base_url"] + "/run.log", 120).decode(errors="replace"))
    return 0


def cmd_pull(args):
    pull(load_state()["base_url"], args.dir)
    return 0


def cmd_kill(_args):
    return 0 if terminate(load_state()["id"]) else 1


def cmd_reap(_args):
    pods = gql("query { myself { pods { id name desiredStatus } } }")["myself"]["pods"]
    if not pods:
        print("no pods running")
        return 0
    for p in pods:
        print(f"{p['id']} {p['name']} {p['desiredStatus']}")
        terminate(p["id"])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def gpu_args(p):
        p.add_argument("--gpu", default=os.environ.get("GPU_TYPE", DEFAULT_GPU))
        # Community hosts are roughly half the price and rather less reliable,
        # which is what the boot timeout and host rotation are for.
        p.add_argument("--cloud", default=os.environ.get("CLOUD_TYPE", "COMMUNITY"),
                       choices=["COMMUNITY", "SECURE", "ALL"])
        return p

    p = gpu_args(sub.add_parser("run"))
    p.add_argument("--out", default="results")
    p.add_argument("--attempts", type=int, default=4)
    p.set_defaults(fn=cmd_run)

    gpu_args(sub.add_parser("create")).set_defaults(fn=cmd_create)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("log").set_defaults(fn=cmd_log)
    p = sub.add_parser("pull")
    p.add_argument("dir")
    p.set_defaults(fn=cmd_pull)
    sub.add_parser("kill").set_defaults(fn=cmd_kill)
    sub.add_parser("reap").set_defaults(fn=cmd_reap)

    args = ap.parse_args()
    if args.cmd == "run":
        # `watch` writes into this directory; keep it on the pod record.
        os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RunPodError as e:
        raise SystemExit(f"RunPod API error:\n{e}")
