"""Quantitative benchmark: render throughput, zero-copy vs CPU-readback.

Both paths render the SAME PBR scene and produce a torch.cuda tensor ready for a
learner. We measure frames/sec of "render -> tensor on GPU":

  - warp_zerocopy : mujofil-warp WarpRenderer.render() -> torch (stays on GPU)
  - mujofil_cpu   : mujofil render_rgb() -> numpy -> torch.from_numpy().cuda()

The difference isolates the cost of the GPU->CPU->GPU round-trip that the
zero-copy path eliminates. Each renderer runs in its OWN subprocess (two Filament
engines can't safely share one process).

Driver:  python bench_quantitative.py
Worker:  python bench_quantitative.py --worker <name> --res <N> --iters <K>
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

IBL_DIR = os.path.join(HERE, "assets", "ibl", "warehouse_new")

SCENE = """
<mujoco>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="copper" rgba="0.95 0.55 0.35 1" metallic="1.0" roughness="0.35"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
    <material name="floor"  rgba="0.45 0.46 0.50 1" metallic="0.0" roughness="0.6"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"/>
    <body pos="-0.6 0 0.30"><freejoint/><geom type="sphere" size="0.26" material="chrome"/></body>
    <body pos="0.1 0 0.30"><freejoint/><geom type="sphere" size="0.26" material="gold"/></body>
    <body pos="0.8 0 0.30"><freejoint/><geom type="box" size="0.24 0.24 0.24" material="copper"/></body>
    <body pos="-0.2 0.7 0.30"><freejoint/><geom type="sphere" size="0.26" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.1" xyaxes="1 0 0 0 0.34 0.94"/>
  </worldbody>
</mujoco>
"""


def _ibl():
    return (os.path.join(IBL_DIR, "warehouse_new_ibl.ktx"),
            os.path.join(IBL_DIR, "warehouse_new_skybox.ktx"))


def worker_warp(res, iters, warmup):
    import mujoco
    import torch
    from mujofil import WarpRenderer, RendererConfig

    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True
    cfg.exposure = 1.4
    r = WarpRenderer(cfg)
    r.load_model(m)
    ibl, sky = _ibl()
    if os.path.exists(ibl):
        r.load_ibl(ibl, sky); r.set_ambient_intensity(9000.0)
    r.add_directional_light(-0.3, 0.2, -1.0, 1.0, 0.98, 0.95, 60000.0, True)
    r.sync_transforms(m, d); r.sync_camera(m, d, 0)

    def step():
        t = r.render()              # (H,W,4) uint8 torch.cuda — zero-copy
        x = t[..., :3].float()      # a typical first op a learner would do
        torch.cuda.synchronize()
        return x

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    dt = time.perf_counter() - t0
    return iters / dt


def worker_cpu(res, iters, warmup):
    import numpy as np
    import mujoco
    import torch
    from mujofil import native as vf

    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    c = vf.RendererConfig()
    c.width = c.height = res
    c.use_vulkan = False          # OpenGL: mujofil's fastest readback path
    c.enable_ssao = True; c.enable_shadows = True; c.enable_msaa = True
    c.exposure = 1.4; c.vsync = False
    r = vf.Renderer(c); r.initialize()
    b = vf.SceneBridge(r); b.load_model(m._address)
    ibl, sky = _ibl()
    if os.path.exists(ibl):
        b.load_ibl(ibl, sky); b.set_ambient_intensity(9000.0)
    b.add_directional_light(-0.3, 0.2, -1.0, 1.0, 0.98, 0.95, 60000.0, True)
    b.sync_transforms(m._address, d._address); b.sync_camera(m._address, d._address, 0)

    dev = "cuda"

    def step():
        rgb = r.render_rgb()                       # (H,W,3) uint8 numpy (CPU)
        x = torch.from_numpy(rgb).to(dev).float()  # upload CPU->GPU
        torch.cuda.synchronize()
        return x

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    dt = time.perf_counter() - t0
    return iters / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["warp", "cpu"], default=None)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    args = ap.parse_args()

    if args.worker == "warp":
        print("FPS " + json.dumps({"fps": worker_warp(args.res, args.iters, args.warmup)}))
        return
    if args.worker == "cpu":
        print("FPS " + json.dumps({"fps": worker_cpu(args.res, args.iters, args.warmup)}))
        return

    # driver
    env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1")
    resolutions = [128, 256, 512]
    print(f"\n{'res':>6} {'mujofil_cpu':>13} {'warp_zerocopy':>15} {'speedup':>9}")
    print("-" * 48)
    for res in resolutions:
        out = {}
        for name in ["cpu", "warp"]:
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--worker", name, "--res", str(res),
                   "--iters", "200", "--warmup", "40"]
            fps = float("nan")
            for _attempt in range(2):  # one retry (occasional transient GPU panic)
                r = subprocess.run(cmd, env=env, capture_output=True, text=True)
                line = [l for l in r.stdout.splitlines() if l.startswith("FPS ")]
                if line:
                    fps = json.loads(line[0][4:])["fps"]
                    break
            if fps != fps:  # NaN -> both attempts failed
                sys.stderr.write(f"[{name}@{res}] failed twice\n")
            out[name] = fps
        sp = out["warp"] / out["cpu"] if out["cpu"] else float("nan")
        print(f"{res:>6} {out['cpu']:>13.0f} {out['warp']:>15.0f} {sp:>8.2f}x")
    print("\n(fps = render -> torch.cuda tensor ready; higher is better)")


if __name__ == "__main__":
    main()
