"""mujofil-warp the way it's MEANT to run: MJWarp GPU physics -> Filament PBR.

MJWarp steps physics on the GPU for N worlds. We read each world's geom
transforms (geom_xpos/geom_xmat) — the small per-frame transform data — and feed
them to mujofil-warp's Filament PBR renderer (Vulkan zero-copy), getting an
(N,H,W,4) torch.cuda tensor.

This is the full pipeline: GPU physics (MJWarp) + photoreal render (Filament) +
zero-copy obs (torch). Reports env-steps/sec (sim + render) and the sim/render
split. Run from /tmp etc; uses the installed mujofil-warp.

Usage: python benchmarks/bench_mjwarp_render.py [--res R --n N --steps K]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
import torch

from mujofil_warp import WarpRenderer, RendererConfig

WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
A = os.path.join(WH, "assets")

SCENE = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    <body pos="-0.6 0 0.5"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body pos="0.1 0 0.6"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
    <body pos="0.8 0.3 0.5"><freejoint/><geom type="sphere" size="0.22" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--warehouse", action="store_true")
    args = ap.parse_args()
    res, N = args.res, args.n

    mjm = mujoco.MjModel.from_xml_string(SCENE)

    # --- MJWarp GPU physics ---
    M = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=N)
    # perturb each world so they differ
    qpos = d.qpos.numpy()
    rng = np.random.default_rng(0)
    qpos[:] += rng.uniform(-0.05, 0.05, size=qpos.shape)
    wp.copy(d.qpos, wp.array(qpos, dtype=float))

    # --- host MjData pool: receives MJWarp transforms, read by the renderer ---
    host = [mujoco.MjData(mjm) for _ in range(N)]
    for h in host:
        mujoco.mj_forward(mjm, h)          # sets camera (cam_xpos/xmat) once
    ngeom = mjm.ngeom

    # --- mujofil-warp Vulkan PBR renderer ---
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = N
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.6
    r = WarpRenderer(cfg); r.load_model(mjm)
    if args.warehouse:
        r.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
                   os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
        for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
            r.load_glb(os.path.join(A, g))
        r.set_ambient_intensity(9000.0)
        for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
            r.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)
    else:
        r.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
                   os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
        r.set_ambient_intensity(9000.0)
        r.add_directional_light(-0.3, 0.2, -1.0, 1.0, 0.98, 0.95, 60000.0, True)

    def one_step():
        # 1) GPU physics
        t = time.perf_counter()
        mjw.step(M, d)
        wp.synchronize()
        # pull MJWarp transforms (GPU->CPU, small: N*ngeom*12 floats)
        gx = d.geom_xpos.numpy()                 # (N, ngeom, 3)
        gm = d.geom_xmat.numpy().reshape(N, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]
            h.geom_xmat[:] = gm[i]
        t_sim = time.perf_counter() - t
        # 2) Filament PBR render (Vulkan zero-copy) -> torch.cuda
        t = time.perf_counter()
        obs = r.render_batch(mjm, host, cam_id=0)   # (N,H,W,4) cuda
        x = obs[..., :3].float()
        torch.cuda.synchronize()
        t_ren = time.perf_counter() - t
        return t_sim, t_ren

    for _ in range(args.warmup):
        one_step()
    sim = ren = 0.0
    t0 = time.perf_counter()
    for _ in range(args.steps):
        s, rr = one_step()
        sim += s; ren += rr
    wall = time.perf_counter() - t0

    sps = N * args.steps / wall
    print(f"MJWARP-PHYSICS + mujofil-warp PBR (Vulkan): "
          f"{res}x{res} N={N} -> {sps:.0f} env-steps/s "
          f"| sim={100*sim/(sim+ren):.0f}% render={100*ren/(sim+ren):.0f}% "
          f"| wall/step={wall/args.steps*1e3:.2f}ms"
          f"{' [WAREHOUSE]' if args.warehouse else ''}")


if __name__ == "__main__":
    main()
