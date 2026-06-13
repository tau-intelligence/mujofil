"""Baseline smoke test for MJWarp GPU physics + its built-in batch renderer.

Goal: prove (1) batched GPU physics steps, (2) the low-fidelity raycaster
renders RGB across N worlds, and (3) the rendered pixels live in Warp GPU
arrays. This establishes the baseline that Option B (Filament PBR) must beat
on fidelity while reusing MJWarp's GPU-resident state.

Run:
    python scripts/smoke_mjwarp.py --nworld 16 --res 256 --out out/
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp


SCENE_XML = """
<mujoco model="smoke">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.5 0.5 0.55 1"/>
    <body name="b0" pos="0 0 0.6">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" rgba="0.8 0.3 0.2 1"/>
    </body>
    <body name="b1" pos="0.8 0.2 0.8">
      <freejoint/>
      <geom type="sphere" size="0.2" rgba="0.3 0.7 0.4 1"/>
    </body>
    <body name="b2" pos="-0.6 -0.4 0.7">
      <freejoint/>
      <geom type="capsule" size="0.12 0.25" rgba="0.3 0.4 0.8 1"/>
    </body>
    <camera name="cam0" pos="2.4 -2.4 1.8" xyaxes="0.7 0.7 0 -0.4 0.4 0.8"/>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wp.init()
    W = H = args.res
    N = args.nworld

    # --- host model, then push to GPU ---
    mjm = mujoco.MjModel.from_xml_string(SCENE_XML)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=N)

    # randomize per-world initial state a little so worlds differ visibly
    qpos = d.qpos.numpy()
    rng = np.random.default_rng(0)
    qpos[:, :] += rng.uniform(-0.05, 0.05, size=qpos.shape)
    wp.copy(d.qpos, wp.array(qpos, dtype=float))

    # --- batched GPU physics ---
    # Warmup: first call JIT-compiles kernels (one-time, cached). Then capture a
    # CUDA graph so timing reflects real steady-state throughput, not compile.
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        mjw.step(m, d)
    graph = cap.graph
    # timed
    t0 = time.perf_counter()
    for _ in range(args.steps):
        wp.capture_launch(graph)
    wp.synchronize()
    t_sim = time.perf_counter() - t0
    sim_sps = N * args.steps / t_sim
    print(f"[physics] {N} worlds x {args.steps} steps in {t_sim*1e3:.1f} ms "
          f"-> {sim_sps:,.0f} env-steps/s (CUDA-graph, post-compile)")

    # --- batch render (low-fi raycaster) ---
    # API differs slightly across versions; try the documented stable form.
    rc = mjw.create_render_context(
        mjm,
        nworld=N,
        cam_res=(W, H),
        render_rgb=True,
        render_depth=True,
        use_textures=True,
        use_shadows=False,
    )

    # warmup (kernel compile) then timed
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()

    t0 = time.perf_counter()
    REN = 30
    for _ in range(REN):
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
    wp.synchronize()
    t_ren = time.perf_counter() - t0
    ren_fps = N * REN / t_ren
    print(f"[render ] {N} worlds @ {W}x{H} x {REN} frames in {t_ren*1e3:.1f} ms "
          f"-> {ren_fps:,.0f} cam-frames/s")

    # --- pull a few images to disk to eyeball fidelity ---
    try:
        from PIL import Image
        rgb = wp.zeros((N, H, W), dtype=wp.vec3)
        mjw.get_rgb(rc, 0, rgb)
        arr = rgb.numpy()  # (N,H,W,3) float in [0,1] (or 0..255?)
        if arr.max() <= 1.0 + 1e-3:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        for i in range(min(N, 4)):
            Image.fromarray(arr[i]).save(os.path.join(args.out, f"mjwarp_world{i}.png"))
        print(f"[save   ] wrote {min(N,4)} PNGs to {args.out}/ "
              f"(rgb array shape={arr.shape}, dtype reports GPU-resident wp.array)")
    except Exception as e:  # noqa: BLE001
        print(f"[save   ] could not save PNGs: {e!r}")

    print("\nSUMMARY")
    print(f"  warp={wp.__version__} mujoco_warp={mjw.__version__} mujoco={mujoco.__version__}")
    print(f"  device={wp.get_device()}")
    print(f"  sim={sim_sps:,.0f} env-steps/s   render={ren_fps:,.0f} cam-frames/s")
    print("  NOTE: MJWarp renderer is single-hit Lambertian (no PBR/IBL/GI).")


if __name__ == "__main__":
    main()
