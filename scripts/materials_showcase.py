"""Materials showcase: where Filament PBR clearly beats MJWarp's flat raycaster.

A row of spheres spanning metallic/roughness, lit by an environment map. The
metallic + low-roughness spheres REFLECT the environment (true PBR), which a
single-hit Lambertian raycaster fundamentally cannot reproduce. Renders the SAME
scene with both renderers for an honest A/B.

Run:
    python scripts/materials_showcase.py --res 768 --out out
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

from mujofil import native as vf

HERE = os.path.dirname(os.path.abspath(__file__))
IBL_ROOT = os.path.join(HERE, os.pardir, "assets", "ibl")


def ibl_paths(name):
    d = os.path.normpath(os.path.join(IBL_ROOT, name))
    # cmgen names files <deployname>_ibl.ktx / <deployname>_skybox.ktx
    base = os.path.basename(d)
    ibl = os.path.join(d, f"{base}_ibl.ktx")
    sky = os.path.join(d, f"{base}_skybox.ktx")
    return ibl, sky


def build_xml():
    """5 spheres: metal->dielectric, each with a different roughness, + a
    glossy floor. Materials carry real metallic/roughness so PBR can shine."""
    mats, geoms = [], []
    # (name, rgba, metallic, roughness)
    specs = [
        ("chrome",  "0.95 0.95 0.97 1", 1.0, 0.05),
        ("gold",    "1.0 0.78 0.34 1",  1.0, 0.18),
        ("copper",  "0.95 0.55 0.35 1", 1.0, 0.35),
        ("plastic", "0.20 0.45 0.85 1", 0.0, 0.25),
        ("rubber",  "0.85 0.25 0.22 1", 0.0, 0.85),
    ]
    n = len(specs)
    for i, (name, rgba, met, rough) in enumerate(specs):
        mats.append(
            f'<material name="{name}" rgba="{rgba}" metallic="{met}" roughness="{rough}"/>')
        x = (i - (n - 1) / 2.0) * 0.62
        geoms.append(
            f'<body name="b{i}" pos="{x:.3f} 0 0.55"><freejoint/>'
            f'<geom type="sphere" size="0.26" material="{name}"/></body>')
    # a slightly glossy floor material so it picks up soft reflections
    floor_mat = '<material name="floor" rgba="0.30 0.31 0.34 1" metallic="0.0" roughness="0.55"/>'
    return f"""
<mujoco model="showcase">
  <visual><headlight diffuse="0.3 0.3 0.3" ambient="0.2 0.2 0.2"/></visual>
  <asset>
    {floor_mat}
    {''.join(mats)}
  </asset>
  <worldbody>
    <light name="key" pos="2 -2 3.5" dir="-0.5 0.5 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor"/>
    {''.join(geoms)}
    <camera name="cam0" pos="0 -3.2 1.25" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>"""


def render_mjwarp(mjm, W, H, steps):
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1)
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    for _ in range(steps):
        mjw.step(m, d)
    wp.synchronize()
    rc = mjw.create_render_context(
        mjm, nworld=1, cam_res=(W, H), render_rgb=True, render_depth=False,
        use_textures=True, use_shadows=True)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    rgb = wp.zeros((1, H, W), dtype=wp.vec3)
    mjw.get_rgb(rc, 0, rgb)
    arr = rgb.numpy()[0]
    if arr.max() <= 1.0 + 1e-3:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, d.qpos.numpy()[0]


def render_pbr(mjm, qpos, W, H, ibl_name):
    host = mujoco.MjData(mjm)
    host.qpos[:] = qpos
    mujoco.mj_forward(mjm, host)

    cfg = vf.RendererConfig()
    cfg.width, cfg.height = W, H
    cfg.use_vulkan = False
    cfg.headless = True
    cfg.vsync = False
    cfg.enable_msaa = True
    cfg.msaa_samples = 4
    cfg.enable_ssao = True
    cfg.enable_shadows = True
    cfg.enable_bloom = True
    cfg.tone_mapping = True
    cfg.render_scale = 1.5  # supersample for crisp reflections

    r = vf.Renderer(cfg)
    if not r.initialize():
        raise RuntimeError("init failed")
    b = vf.SceneBridge(r)
    b.load_model(mjm._address)
    ibl, sky = ibl_paths(ibl_name)
    b.load_ibl(ibl, sky)
    b.set_ambient_intensity(1.0)
    b.add_directional_light(-0.5, 0.5, -1.0, 1.0, 0.98, 0.95, 70000.0, True)
    b.sync_transforms(mjm._address, host._address)
    b.sync_camera(mjm._address, host._address, 0)
    img = np.asarray(r.render_rgb())
    # NOTE: skip r.destroy() (teardown double-free); images already captured.
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--ibl", type=str, default="warehouse_new")
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = H = args.res

    mjm = mujoco.MjModel.from_xml_string(build_xml())
    from PIL import Image

    print("[1] MJWarp flat raycaster ...")
    flat, qpos = render_mjwarp(mjm, W, H, args.steps)
    Image.fromarray(flat).save(os.path.join(args.out, "showcase_flat.png"))

    print("[2] Filament PBR (same state) ...")
    pbr = render_pbr(mjm, qpos, W, H, args.ibl)
    Image.fromarray(pbr).save(os.path.join(args.out, "showcase_pbr.png"))

    # side-by-side strip
    h = min(flat.shape[0], pbr.shape[0])
    strip = np.concatenate([flat[:h], pbr[:h]], axis=1)
    Image.fromarray(strip).save(os.path.join(args.out, "showcase_ab.png"))

    print(f"    flat mean={flat.mean():.1f} std={flat.std():.1f} | "
          f"pbr mean={pbr.mean():.1f} std={pbr.std():.1f}")
    print("DONE -> out/showcase_flat.png, out/showcase_pbr.png, out/showcase_ab.png")


if __name__ == "__main__":
    main()
