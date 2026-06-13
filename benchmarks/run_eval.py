"""Consolidated evaluation: ours vs stock MuJoCo vs MJWarp, all -> torch.cuda.

Runs every (renderer, resolution, N) cell in its own subprocess (isolation),
retries once on transient failure, and prints one consolidated table.

Renderers (all produce camera images delivered to torch on the GPU):
  - mujoco     : stock CPU MuJoCo EGL renderer, N envs sequential        (flat)
  - mjwarp     : MJWarp built-in raycaster, batched N worlds             (flat)
  - ours       : mujofil-warp PBR, batched N worlds, zero-copy           (PBR)

Run:  python benchmarks/run_eval.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "bench_vs_mjwarp.py")

ENV = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1", MUJOCO_GL="egl")

# (label, worker-name)
RENDERERS = [("mujoco (flat)", "mujoco"),
             ("mjwarp (flat)", "mjwarp"),
             ("ours  (PBR)", "pbr_batch")]
RES = [128, 256, 512]
NS = [1, 16, 64, 256]


def run(worker, res, n, iters, warmup):
    cmd = [sys.executable, WORKER, "--worker", worker, "--res", str(res),
           "--iters", str(iters), "--warmup", str(warmup), "--nworld", str(n)]
    for _ in range(2):
        r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("FPS ")]
        if line:
            return json.loads(line[0][4:])["fps"]
    return float("nan")


def iters_for(res, n):
    base = 60 if res <= 128 else (40 if res <= 256 else 25)
    if n >= 64:
        base = max(15, base // 2)
    return base, max(5, base // 6)


def main():
    print("\nThroughput: rendered camera-images / second (higher = better).")
    print("All renderers deliver images to a torch.cuda tensor.\n")
    for res in RES:
        print(f"=== resolution {res}x{res} ===")
        print(f"{'renderer':>14} " + " ".join(f"{'N='+str(n):>9}" for n in NS))
        print("-" * (15 + 10 * len(NS)))
        for label, worker in RENDERERS:
            cells = []
            for n in NS:
                it, wu = iters_for(res, n)
                cells.append(run(worker, res, n, it, wu))
            row = " ".join((f"{c:>9.0f}" if c == c else f"{'--':>9}") for c in cells)
            print(f"{label:>14} {row}")
        print()
    print("Notes: mujoco/mjwarp = flat shading; ours = full PBR+IBL+shadows.")
    print("       mujoco renders N sequentially; mjwarp & ours batch N worlds.")


if __name__ == "__main__":
    main()
