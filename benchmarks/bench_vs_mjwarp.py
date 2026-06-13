"""MJWarp-renderer vs mujofil-warp (ours) on the SAME scene.

Honest head-to-head of the two renderers, both producing torch.cuda tensors:

  - mjwarp_raycaster : MJWarp's built-in single-hit raycaster (flat Lambertian,
                       GPU-resident). Measured at nworld=1 (per-camera, fair to
                       our single render) AND batched (its real strength).
  - warp_pbr         : ours — Filament PBR (reflections/IBL/shadows) zero-copy.

Same MuJoCo scene, same camera, same resolution. This is the key comparison:
MJWarp trades fidelity for throughput; we add photoreal PBR. The numbers show
exactly what that fidelity costs.

Each renderer runs in its own subprocess.
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

# A primitives-only scene (MJWarp prefers primitives; identical geometry for both).
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


def worker_pbr(res, iters, warmup):
    import mujoco
    import torch
    from mujofil_warp import WarpRenderer, RendererConfig

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
        t = r.render()[..., :3].float()
        torch.cuda.synchronize()
        return t

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return iters / (time.perf_counter() - t0)


def worker_pbr_batch(res, iters, warmup, nworld):
    """Ours, BATCHED: render nworld worlds per call, one GPU sync. cameras/sec."""
    import numpy as np
    import mujoco
    import torch
    from mujofil_warp import WarpRenderer, RendererConfig

    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(nworld)]
    for i, dd in enumerate(datas):
        dd.qpos[:] = 0
        # nudge each world so they differ (realistic: distinct states)
        if m.nq >= 1:
            dd.qpos[0] += 0.01 * i
        mujoco.mj_forward(m, dd)

    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.batch_size = nworld
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True
    cfg.exposure = 1.4
    r = WarpRenderer(cfg)
    r.load_model(m)
    ibl, sky = _ibl()
    if os.path.exists(ibl):
        r.load_ibl(ibl, sky); r.set_ambient_intensity(9000.0)
    r.add_directional_light(-0.3, 0.2, -1.0, 1.0, 0.98, 0.95, 60000.0, True)

    def step():
        t = r.render_batch(m, datas, cam_id=0)[..., :3].float()
        torch.cuda.synchronize()
        return t

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    dt = time.perf_counter() - t0
    return nworld * iters / dt   # cameras/sec


def worker_mjwarp(res, iters, warmup, nworld):
    """MJWarp raycaster. Throughput = cameras/sec = nworld / time_per_render."""
    import numpy as np
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    import torch

    mjm = mujoco.MjModel.from_xml_string(SCENE)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()

    rc = mjw.create_render_context(
        mjm, nworld=nworld, cam_res=(res, res), render_rgb=True,
        render_depth=False, use_textures=True, use_shadows=True)
    rgb = wp.zeros((nworld, res, res), dtype=wp.vec3)

    def step():
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        mjw.get_rgb(rc, 0, rgb)
        # to torch (already on GPU via warp->torch interop)
        t = wp.to_torch(rgb)
        torch.cuda.synchronize()
        return t

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    dt = time.perf_counter() - t0
    return nworld * iters / dt   # cameras/sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["pbr", "pbr_batch", "mjwarp"], default=None)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--nworld", type=int, default=1)
    args = ap.parse_args()

    if args.worker == "pbr":
        print("FPS " + json.dumps({"fps": worker_pbr(args.res, args.iters, args.warmup)}))
        return
    if args.worker == "pbr_batch":
        print("FPS " + json.dumps({"fps": worker_pbr_batch(args.res, args.iters, args.warmup, args.nworld)}))
        return
    if args.worker == "mjwarp":
        print("FPS " + json.dumps({"fps": worker_mjwarp(args.res, args.iters, args.warmup, args.nworld)}))
        return

    env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1")

    def run(worker, res, nworld=1, iters=200, warmup=40):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", worker,
               "--res", str(res), "--iters", str(iters), "--warmup", str(warmup),
               "--nworld", str(nworld)]
        for _ in range(2):
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("FPS ")]
            if line:
                return json.loads(line[0][4:])["fps"]
        sys.stderr.write(f"[{worker} res={res} N={nworld}] failed\n")
        return float("nan")

    print("\n=== Single camera (apples-to-apples: 1 render = 1 image) ===")
    print(f"{'res':>6} {'mjwarp_flat':>13} {'warp_pbr(ours)':>16} {'pbr/flat':>9}")
    print("-" * 50)
    for res in [128, 256, 512]:
        flat = run("mjwarp", res, nworld=1)
        pbr = run("pbr", res)
        ratio = pbr / flat if flat == flat and flat else float("nan")
        print(f"{res:>6} {flat:>13.0f} {pbr:>16.0f} {ratio:>8.2f}x")

    print("\n=== MJWarp BATCHED raycaster (its real strength), cameras/sec ===")
    print(f"{'res':>6} {'N=1':>10} {'N=16':>10} {'N=64':>10} {'N=256':>10}")
    print("-" * 52)
    for res in [128, 256]:
        cols = [run("mjwarp", res, nworld=n, iters=80, warmup=10) for n in [1, 16, 64, 256]]
        print(f"{res:>6} " + " ".join(f"{c:>10.0f}" for c in cols))

    print("\n=== OURS BATCHED PBR (apples-to-apples), cameras/sec ===")
    print(f"{'res':>6} {'N=1':>10} {'N=16':>10} {'N=64':>10} {'N=256':>10}")
    print("-" * 52)
    for res in [128, 256]:
        cols = [run("pbr_batch", res, nworld=n, iters=60, warmup=10) for n in [1, 16, 64, 256]]
        print(f"{res:>6} " + " ".join(f"{c:>10.0f}" for c in cols))

    print("\nNotes:")
    print(" - mjwarp_flat = single-hit Lambertian (no PBR/IBL/reflections).")
    print(" - ours = full PBR + IBL + shadows, zero-copy to torch.")
    print(" - cameras/sec = total images/sec (N worlds per batched call).")


if __name__ == "__main__":
    main()
