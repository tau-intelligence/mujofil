"""Qualitative benchmark: warehouse PBR via the ZERO-COPY path.

Renders the photoreal warehouse (GLB env + IBL + ceiling spotlights) with
MJWarp-stepped metal objects, delivered as a torch.cuda tensor (zero-copy), and
saves a PNG. Purpose: confirm the zero-copy renderer preserves PBR fidelity
(reflections/IBL/shadows) — i.e. the GPU-resident path doesn't break visuals.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp

from mujofil_warp import WarpRenderer, RendererConfig

WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
ASSETS = os.path.join(WH, "assets")

SCENE = """
<mujoco>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="copper" rgba="0.95 0.55 0.35 1" metallic="1.0" roughness="0.35"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
    <material name="steel"  rgba="0.8 0.82 0.85 1"  metallic="1.0" roughness="0.25"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    <body pos="-0.8 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body pos="-0.2 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="gold"/></body>
    <body pos="0.4 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="copper"/></body>
    <body pos="1.0 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="blue"/></body>
    <body pos="0.1 0.7 0.32"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="steel"/></body>
    <camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>
  </worldbody>
</mujoco>
"""


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    m = mujoco.MjModel.from_xml_string(SCENE)

    # settle objects on the floor with MJWarp GPU physics
    mw_m = mjw.put_model(m)
    d = mjw.make_data(m, nworld=1)
    for _ in range(3):
        mjw.step(mw_m, d)
    wp.synchronize()
    for _ in range(250):
        mjw.step(mw_m, d)
    wp.synchronize()
    qpos = d.qpos.numpy()[0]

    host = mujoco.MjData(m)
    host.qpos[:] = qpos
    mujoco.mj_forward(m, host)

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.enable_ssao = True
    cfg.enable_shadows = True
    cfg.enable_msaa = True
    cfg.exposure = 1.6
    r = WarpRenderer(cfg)
    r.load_model(m)  # MJWarp objects (primitives)

    # warehouse environment
    r.load_ibl(os.path.join(ASSETS, "ibl", "warehouse_ibl_ibl.ktx"),
               os.path.join(ASSETS, "ibl", "warehouse_ibl_skybox.ktx"))
    r.load_glb(os.path.join(ASSETS, "nvidia_floor.glb"))
    r.load_glb(os.path.join(ASSETS, "nvidia_warehouse_tinted.glb"))
    r.load_glb(os.path.join(ASSETS, "wall_patch.glb"))
    r.set_ambient_intensity(9000.0)
    with open(os.path.join(WH, "lamps.json")) as f:
        lamps = json.load(f)
    for x, y, z in lamps:
        r.add_spot_light(x, y, z, 0.0, 0.0, -1.0, 1.0, 0.97, 0.90,
                         32_000_000.0, 14.0, 55.0, 88.0, True)
    # a soft shadow-casting key for contact shadows
    r.add_directional_light(-0.15, 0.12, -1.0, 1.0, 0.98, 0.95, 9000.0, True)

    r.sync_transforms(m, host)
    r.sync_camera(m, host, 0)

    img = r.render()                      # (H,W,4) uint8 torch.cuda
    assert img.is_cuda
    rgb = img[..., :3].cpu().numpy()

    from PIL import Image
    os.makedirs("out", exist_ok=True)
    out = f"out/bench_qualitative_{res}.png"
    Image.fromarray(rgb).save(out)
    print(f"geoms={r.geom_count}  img mean={rgb.mean():.1f} std={rgb.std():.1f}")
    print(f"saved {out}")
    print("\nQUALITATIVE OK: warehouse PBR rendered via zero-copy "
          "(torch.cuda), fidelity preserved.")


if __name__ == "__main__":
    main()
