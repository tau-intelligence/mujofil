"""Minimal mujofil-warp example: MJWarp GPU physics -> Filament PBR -> torch.cuda.

Demonstrates the quality toggles / presets so you can reproduce the
fidelity-vs-throughput trends from ``benchmarks/`` on your own hardware.

Run::

    # full photoreal PBR (SSAO on)
    MUJOFIL_WARP_BACKEND=gl python examples/minimal_render.py --preset high

    # ~2x faster (SSAO off) -- the single biggest throughput lever
    MUJOFIL_WARP_BACKEND=gl python examples/minimal_render.py --preset fast

The GL backend (MUJOFIL_WARP_BACKEND=gl) does a true single-sync batch and needs
an X display. The default Vulkan backend uses a shared device + exportable
swapchain. Both deliver frames as zero-copy torch.cuda tensors.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco_warp as mjw
import warp as wp
import torch

from mujofil_warp import WarpRenderer, QUALITY_PRESETS

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

    mjm = mujoco.MjModel.from_xml_string(SCENE)
    M = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=args.n)
    host = [mujoco.MjData(mjm) for _ in range(args.n)]
    for h in host:
        mujoco.mj_forward(mjm, h)
    ngeom = mjm.ngeom

    # All quality toggles flow from the preset; override any individually, e.g.
    # WarpRenderer(..., preset="high", ssao=False).
    r = WarpRenderer(width=args.res, height=args.res, batch_size=args.n, preset=args.preset)
    r.load_model(mjm)

    def step():
        mjw.step(M, d)
        wp.synchronize()
        gx = d.geom_xpos.numpy()
        gm = d.geom_xmat.numpy().reshape(args.n, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]
            h.geom_xmat[:] = gm[i]
        obs = r.render_batch(mjm, host, cam_id=0)  # (N, H, W, 4) uint8 torch.cuda
        torch.cuda.synchronize()
        return obs

    for _ in range(5):  # warmup (first MJWarp call JIT-compiles)
        step()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        obs = step()
    dt = time.perf_counter() - t0

    cam_s = args.n * args.steps / dt
    print(f"preset={args.preset} res={args.res} N={args.n}: {cam_s:.0f} cam/s "
          f"| obs {tuple(obs.shape)} {obs.dtype} on {obs.device} (zero-copy)")

    if args.save:
        from PIL import Image
        Image.fromarray(obs[0, ..., :3].cpu().numpy()).save(args.save)
        print(f"saved world-0 frame -> {args.save}")


if __name__ == "__main__":
    main()
