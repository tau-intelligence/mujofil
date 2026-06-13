"""Phase 1 — naive correctness bridge: MJWarp state -> Filament PBR.

Proves the data mapping works and shows the fidelity payoff, ignoring
performance (this version DOES round-trip through the CPU; Phase 2 removes that).

Pipeline per shown world:
  1. MJWarp steps physics for N worlds on GPU (mjw.step).
  2. MJWarp renders world i with its built-in single-hit raycaster  -> *_flat.png
  3. We copy world i's qpos GPU->CPU, set it on a host mujoco.MjData, run
     mj_forward (recomputes geom_xpos/xmat the standard way, so the geom layout
     matches what mujofil's SceneBridge expects), then render with Filament's
     PBR renderer (same camera)                                    -> *_pbr.png

Run:
    python scripts/phase1_bridge.py --nworld 4 --res 512 --steps 80 --out out
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

from mujofil import native as vf

HERE = os.path.dirname(os.path.abspath(__file__))
IBL_DIR = os.path.join(HERE, os.pardir, "assets", "ibl", "studio")
IBL_KTX = os.path.normpath(os.path.join(IBL_DIR, "studio_ibl_ibl.ktx"))
SKYBOX_KTX = os.path.normpath(os.path.join(IBL_DIR, "studio_ibl_skybox.ktx"))


# A scene of primitives with distinct materials. PBR (metal/rough + specular)
# vs MJWarp's flat Lambertian should be visually obvious here.
SCENE_XML = """
<mujoco model="phase1">
  <visual>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.2 0.2 0.2"/>
  </visual>
  <worldbody>
    <light name="key" pos="1.5 -1.5 3" dir="-0.4 0.4 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.78 0.76 0.72 1"/>
    <body name="box" pos="0 0 0.45">
      <freejoint/>
      <geom type="box" size="0.22 0.22 0.22" rgba="0.85 0.25 0.2 1"/>
    </body>
    <body name="ball" pos="0.7 0.25 0.6">
      <freejoint/>
      <geom type="sphere" size="0.22" rgba="0.2 0.55 0.85 1"/>
    </body>
    <body name="pin" pos="-0.6 -0.4 0.55">
      <freejoint/>
      <geom type="capsule" size="0.13 0.26" rgba="0.85 0.75 0.2 1"/>
    </body>
    <camera name="cam0" pos="2.4 -2.4 1.4" xyaxes="0.7071 0.7071 0 -0.227 0.227 0.947"/>
  </worldbody>
</mujoco>
"""

FILL = 0  # placeholder


def render_mjwarp(mjm, N, W, H, steps, out):
    """Step physics on GPU + render world 0..k with MJWarp's raycaster."""
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=N)

    # perturb per-world so the worlds diverge visibly
    qpos = d.qpos.numpy()
    rng = np.random.default_rng(0)
    qpos[:] += rng.uniform(-0.04, 0.04, size=qpos.shape)
    wp.copy(d.qpos, wp.array(qpos, dtype=float))

    # warmup compile, then step
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    for _ in range(steps):
        mjw.step(m, d)
    wp.synchronize()

    rc = mjw.create_render_context(
        mjm, nworld=N, cam_res=(W, H),
        render_rgb=True, render_depth=False,
        use_textures=True, use_shadows=True)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()

    rgb = wp.zeros((N, H, W), dtype=wp.vec3)
    mjw.get_rgb(rc, 0, rgb)
    arr = rgb.numpy()
    if arr.max() <= 1.0 + 1e-3:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # return both the images AND the evolved qpos so the PBR render matches
    return arr, d.qpos.numpy()


def render_pbr(mjm, qpos_all, W, H, out, k):
    """Render the SAME states with Filament PBR via a host MjData."""
    host = mujoco.MjData(mjm)

    cfg = vf.RendererConfig()
    cfg.width, cfg.height = W, H
    cfg.use_vulkan = False          # readback path here -> OpenGL (Phase 1 only)
    cfg.headless = True
    cfg.vsync = False
    cfg.enable_msaa = True
    cfg.msaa_samples = 4
    cfg.enable_ssao = True
    cfg.enable_shadows = True
    cfg.enable_bloom = True
    cfg.tone_mapping = True
    cfg.render_scale = 1.0

    r = vf.Renderer(cfg)
    if not r.initialize():
        raise RuntimeError("Filament renderer failed to initialize")
    b = vf.SceneBridge(r)
    b.load_model(mjm._address)

    # Image-based lighting: this is what makes PBR visibly win over flat shading
    # (real reflections, ambient occlusion, grounded look). Skybox doubles as a
    # backdrop instead of a flat clear color.
    if os.path.exists(IBL_KTX) and os.path.exists(SKYBOX_KTX):
        b.load_ibl(IBL_KTX, SKYBOX_KTX)
        b.set_ambient_intensity(1.0)
    else:
        b.set_ambient_intensity(0.35)
        print(f"    [warn] IBL not found at {IBL_DIR}; falling back to flat ambient")

    # one key directional light WITH shadows for contact grounding + crisp form
    b.add_directional_light(-0.4, 0.4, -1.0, 1.0, 0.98, 0.95, 80000.0, True)

    from PIL import Image
    n = min(k, qpos_all.shape[0])
    for i in range(n):
        host.qpos[:] = qpos_all[i]
        mujoco.mj_forward(mjm, host)          # recompute geom_xpos/xmat + camera
        b.sync_transforms(mjm._address, host._address)
        b.sync_camera(mjm._address, host._address, 0)
        img = r.render_rgb()                  # (H,W,3) uint8
        Image.fromarray(np.asarray(img)).save(os.path.join(out, f"pbr_world{i}.png"))
    # NOTE: intentionally NOT calling r.destroy(): the current mujofil teardown
    # double-frees FIndirectLight (LightManager::clear) and aborts. Images are
    # already written; let the process exit cleanly instead.
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=4)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--show", type=int, default=2, help="how many worlds to save")
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = H = args.res

    mjm = mujoco.MjModel.from_xml_string(SCENE_XML)

    print("[1] MJWarp GPU physics + raycaster ...")
    flat, qpos_all = render_mjwarp(mjm, args.nworld, W, H, args.steps, args.out)
    from PIL import Image
    for i in range(min(args.show, flat.shape[0])):
        Image.fromarray(flat[i]).save(os.path.join(args.out, f"flat_world{i}.png"))
    print(f"    saved {min(args.show, flat.shape[0])} flat (raycaster) PNGs")

    print("[2] Filament PBR from the SAME MJWarp states ...")
    n = render_pbr(mjm, qpos_all, W, H, args.out, args.show)
    print(f"    saved {n} PBR PNGs")

    # quick brightness/contrast sanity (not a fidelity metric, just a signal)
    for i in range(n):
        a = np.asarray(Image.open(os.path.join(args.out, f"flat_world{i}.png")).convert("RGB"))
        b = np.asarray(Image.open(os.path.join(args.out, f"pbr_world{i}.png")).convert("RGB"))
        print(f"    world{i}: flat mean={a.mean():.1f} std={a.std():.1f} | "
              f"pbr mean={b.mean():.1f} std={b.std():.1f}")

    print("\nDONE. Compare out/flat_world*.png (MJWarp) vs out/pbr_world*.png (Filament PBR).")
    print("Same physics state, two renderers: flat Lambertian vs full PBR.")


if __name__ == "__main__":
    main()
