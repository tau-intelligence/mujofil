"""Minimal mujofil example: N MuJoCo worlds -> photoreal frames on torch.cuda.

You only import ``mujofil``. ``ParallelScene`` drives MuJoCo Warp (GPU MuJoCo)
for the physics and this package's Filament renderer for the pixels, handing each
batch of observations back as a zero-copy ``torch.cuda`` tensor.

Run::

    # full photoreal PBR (SSAO on) -- uses the default GL backend
    python examples/minimal_render.py --preset high

    # ~2x faster (SSAO off) -- the single biggest throughput lever
    python examples/minimal_render.py --preset fast

The default GL backend does a true single-sync batch and needs an X display
(it auto-falls back to Vulkan when none is available). Force a backend with
MUJOFIL_BACKEND=gl|vulkan. Both deliver frames as zero-copy torch.cuda tensors.
"""
from __future__ import annotations

import argparse
import time

import torch

import mujofil
from mujofil import QUALITY_PRESETS

SCENE = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body pos="-0.6 0 0.6"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body pos="0.1 0 0.7"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
    <body pos="0.8 0.3 0.6"><freejoint/><geom type="sphere" size="0.22" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(QUALITY_PRESETS), default="high",
                    help="quality preset (high=photoreal, fast=SSAO off ~2x, ultra, raw)")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n", type=int, default=32, help="number of parallel worlds")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--save", type=str, default="", help="save world-0 frame to this PNG")
    args = ap.parse_args()

    # One object: GPU physics (MuJoCo Warp) + photoreal rendering (Filament).
    scene = mujofil.ParallelScene(
        SCENE, num_worlds=args.n, width=args.res, height=args.res, preset=args.preset)

    for _ in range(5):  # warmup (the first MuJoCo Warp step JIT-compiles)
        scene.step()
        scene.render(camera="cam0")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        scene.step()
        obs = scene.render(camera="cam0")   # (N, H, W, 4) uint8 torch.cuda, zero-copy
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    cam_s = args.n * args.steps / dt
    print(f"preset={args.preset} res={args.res} N={args.n}: {cam_s:.0f} cam/s "
          f"| obs {tuple(obs.shape)} {obs.dtype} on {obs.device} (zero-copy)")

    if args.save:
        from PIL import Image
        Image.fromarray(obs[0, ..., :3].cpu().numpy()).save(args.save)
        print(f"saved world-0 frame -> {args.save}")

    scene.close()


if __name__ == "__main__":
    main()
