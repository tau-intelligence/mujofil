"""Probe high-N feasibility: does the N-image swapchain + render survive, and
what does it cost? Warehouse, MJWarp physics + Vulkan PBR. Increments N, prints
VRAM and timing, stops on failure."""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0]
        u, t = out.split(",")
        return f"{int(u)}/{int(t)}MiB"
    except Exception:
        return "?"


def run_one(res, n):
    """Run the mjwarp render bench in a subprocess; return env-steps/s or None."""
    cmd = [sys.executable, os.path.join(HERE, "benchmarks", "bench_mjwarp_render.py"),
           "--res", str(res), "--n", str(n), "--steps", "12", "--warmup", "3",
           "--warehouse"]
    env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    wall = time.perf_counter() - t0
    line = [l for l in p.stdout.splitlines() if "MJWARP-PHYSICS" in l]
    if line:
        return line[0], wall
    err = "\n".join(p.stderr.splitlines()[-4:])
    return f"FAILED ({err[:160]})", wall


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    print(f"High-N probe @ {res}x{res} warehouse (VRAM before each run)")
    for n in [64, 128, 256, 512, 1024, 2048]:
        print(f"  [N={n:>4}] VRAM={vram()} ... ", end="", flush=True)
        try:
            res_line, wall = run_one(res, n)
            print(f"({wall:.0f}s)\n         {res_line}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT (>600s) — stopping")
            break


if __name__ == "__main__":
    main()
