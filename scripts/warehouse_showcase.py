"""Warehouse showcase: MJWarp-driven objects inside the photoreal warehouse.

This is the real pitch for mujofil-warp: take objects whose physics is stepped
by MJWarp (GPU), and render them inside the full NVIDIA-style warehouse
environment (PBR floor/walls/racks + image-based lighting + ceiling spotlights).
The rich environment is exactly what makes PBR metals/gloss read as photoreal —
something MJWarp's flat single-hit raycaster cannot do.

Renders the SAME MJWarp state two ways for an honest A/B:
  - flat: MJWarp's built-in raycaster (objects only; no warehouse)
  - pbr : Filament inside the warehouse backdrop

Run:
    python scripts/warehouse_showcase.py --res 1280 --out out
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

# Make the venv's native module importable under the name warehouse_env expects.
from mujofil import native as _vf
sys.modules.setdefault("_vf_mujoco_native", _vf)

WAREHOUSE_DIR = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse"
sys.path.insert(0, WAREHOUSE_DIR)
from warehouse_env import Warehouse  # noqa: E402


def build_xml():
    """Material spheres + a couple boxes, resting on the warehouse floor (z=0).
    The collision floor is group=3 so MJWarp physics works but mujofil does not
    draw it (the warehouse GLB floor is the visible one)."""
    specs = [
        ("chrome",  "0.95 0.95 0.97 1", 1.0, 0.06),
        ("gold",    "1.0 0.78 0.34 1",  1.0, 0.20),
        ("copper",  "0.95 0.55 0.35 1", 1.0, 0.38),
        ("plastic", "0.20 0.45 0.85 1", 0.0, 0.30),
        ("rubber",  "0.85 0.25 0.22 1", 0.0, 0.85),
    ]
    mats, bodies = [], []
    n = len(specs)
    for i, (name, rgba, met, rough) in enumerate(specs):
        mats.append(
            f'<material name="{name}" rgba="{rgba}" metallic="{met}" roughness="{rough}"/>')
        x = (i - (n - 1) / 2.0) * 0.62
        bodies.append(
            f'<body name="b{i}" pos="{x:.3f} 0 0.55"><freejoint/>'
            f'<geom type="sphere" size="0.24" material="{name}"/></body>')
    # a metal box for a hard-surface highlight
    mats.append('<material name="steel" rgba="0.8 0.82 0.85 1" metallic="1.0" roughness="0.25"/>')
    bodies.append('<body name="box" pos="0 0.7 0.6"><freejoint/>'
                  '<geom type="box" size="0.22 0.22 0.22" material="steel"/></body>')
    return f"""
<mujoco model="wh_showcase">
  <asset>
    {''.join(mats)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


def step_mjwarp(mjm, steps):
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1)
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    for _ in range(steps):
        mjw.step(m, d)
    wp.synchronize()
    return d, m


def render_flat(mjm, m, d, W, H, out):
    """MJWarp's own raycaster, objects only (no warehouse env)."""
    rc = mjw.create_render_context(
        mjm, nworld=1, cam_res=(W, H), render_rgb=True, render_depth=False,
        use_textures=True, use_shadows=True)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    rgb = wp.zeros((1, H, W), dtype=wp.vec3)
    mjw.get_rgb(rc, 0, rgb)
    arr = rgb.numpy()[0]
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 + 1e-3 \
        else np.clip(arr, 0, 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(arr).save(os.path.join(out, "wh_flat.png"))
    return arr


def render_pbr(mjm, qpos, W, H, out, camera="floor"):
    """Filament PBR with the full warehouse environment around the objects."""
    host = mujoco.MjData(mjm)
    host.qpos[:] = qpos
    mujoco.mj_forward(mjm, host)

    wh = Warehouse()
    r, b = wh.create_renderer(W, H)
    b.load_model(mjm._address)      # our MJWarp objects (primitives)
    wh.load(b)                      # warehouse IBL + GLB env + ceiling spots
    b.sync_transforms(mjm._address, host._address)
    b.set_free_camera(*wh.CAMERAS[camera])
    img = np.asarray(r.render())[:, :, :3]
    from PIL import Image
    Image.fromarray(img).save(os.path.join(out, "wh_pbr.png"))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--camera", type=str, default="floor")
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = H = args.res

    mjm = mujoco.MjModel.from_xml_string(build_xml())

    print("[1] MJWarp physics ...")
    d, m = step_mjwarp(mjm, args.steps)
    qpos = d.qpos.numpy()[0].copy()

    # PBR warehouse render FIRST (only needs qpos) so an MJWarp-render hiccup
    # can't block the main deliverable.
    print("[2] Filament PBR inside the warehouse ...")
    pbr = render_pbr(mjm, qpos, W, H, args.out, args.camera)
    print(f"    pbr mean={pbr.mean():.1f} std={pbr.std():.1f}  -> out/wh_pbr.png")

    print("[3] MJWarp flat raycaster (reference) ...")
    try:
        flat = render_flat(mjm, m, d, W, H, args.out)
        print(f"    flat mean={flat.mean():.1f} std={flat.std():.1f}  -> out/wh_flat.png")
    except Exception as e:  # noqa: BLE001
        print(f"    [skip] MJWarp flat render failed: {e!r}")

    print("DONE")


if __name__ == "__main__":
    main()
