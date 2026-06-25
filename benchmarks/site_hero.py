"""Fresh high-res hero stills for the MuJoFil website (no old screenshots).

Renders each photoreal GLB environment with the full render_batch PBR path
(Filament PBR + IBL + shadows + in-engine FILMIC tonemap), at website resolution,
from a couple of hand-framed eye-level viewpoints per scene so we can keep the
best. Also renders a clean parallel-batch montage (one scene, N worlds in a
single instanced layered draw) for the "parallel rendering" figure.

Run (uncapped address space for CUDA):
  systemd-run --user -p LimitAS=infinity --quiet --wait --pipe -- \
    bash -c 'cd ~/mujofil-warp && source .venv/bin/activate && \
      PYTHONPATH=$PWD:/home/mumuksh/Visual-Fidelity-Mujoco \
      MUJOFIL_WARP_BACKEND=gl python benchmarks/site_hero.py'
"""
import os
import sys

os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR",
                      os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)
sys.path.insert(0, VFM)

import numpy as np
import mujoco
from PIL import Image
from mujofil import WarpRenderer, RendererConfig
from trailer.scenes import SCENES

OUT = os.path.join(HERE, "site_assets", "img")
os.makedirs(OUT, exist_ok=True)
ENV = os.path.join(VFM, "assets", "envmaps")
EMPTY = "<mujoco><worldbody/></mujoco>"

# Eye-level (eye xyz -> look xyz), RELATIVE to the scene's GLB origin (xform tx,ty).
VIEWS = {
    "sponza": [
        ((-9.5, 0.0, 2.4), (6.0, 0.2, 1.6)),
        ((6.5, -2.0, 2.0), (-6.0, 1.0, 1.8)),
    ],
    "living": [
        ((1.9, 1.7, 1.4), (-0.9, 0.4, 0.6)),
        ((2.5, -1.6, 1.4), (-1.0, 0.6, 0.6)),
    ],
    "cafe": [
        ((-3.0, 2.8, 1.5), (0.6, -0.8, 0.85)),
        ((2.8, 2.4, 1.5), (-1.0, -0.8, 0.85)),
    ],
    "gallery": [
        ((-4.5, 3.5, 1.6), (1.0, -1.0, 1.1)),
        ((3.5, -3.0, 1.6), (-1.0, 1.0, 1.2)),
    ],
}

# Warehouse is a multi-GLB scene (floor + shell + wall patch) lit by ceiling
# spotlights from lamps.json -- rendered via its own path below.
WAREHOUSE_ASSETS = os.path.join(VFM, "warehouse", "data", "assets")
WAREHOUSE_LAMPS = os.path.join(VFM, "warehouse", "data", "lamps.json")
WAREHOUSE_VIEWS = [
    (10.0, -9.0, 1.7, -10.0, 9.0, 2.0),   # hero diagonal
    (6.0, -4.0, 0.9, 0.0, 2.0, 0.6),      # low floor glide
]


def ibl_paths(name):
    d = os.path.join(ENV, name)
    ii = os.path.join(d, f"{name}_ibl.ktx")
    ss = os.path.join(d, f"{name}_skybox.ktx")
    if os.path.exists(ii):
        return ii, ss
    d = os.path.join(HERE, "assets", "ibl", "studio")
    return os.path.join(d, "studio_ibl_ibl.ktx"), os.path.join(d, "studio_ibl_skybox.ktx")


def render_scene(scene, w=1280, h=800):
    s = SCENES[scene]
    xform = [float(x) for x in s["xform"]]
    cx, cy = xform[12], xform[13]
    views = VIEWS[scene]
    n = len(views)
    m = mujoco.MjModel.from_xml_string(EMPTY)
    cfg = RendererConfig()
    cfg.width, cfg.height = w, h
    cfg.batch_size = n
    cfg.enable_shadows = True
    cfg.enable_ssao = True
    cfg.enable_bloom = True
    cfg.exposure = float(s.get("exposure", 0.6))
    r = WarpRenderer(cfg)
    r.load_model(m)
    ii, ss = ibl_paths(s.get("ibl", "studio_ibl"))
    if os.path.exists(ii):
        r.load_ibl(ii, ss)
    r.set_ambient_intensity(float(s.get("ambient", 8000.0)) * 1.5)
    r.load_glb_xform(s["glb"], xform)
    for i, (eye, look) in enumerate(views):
        r.set_free_camera(cx + eye[0], cy + eye[1], eye[2],
                          cx + look[0], cy + look[1], look[2])
        img = r.render()[..., :3].clamp(0, 255).byte().cpu().numpy()
        path = os.path.join(OUT, f"{scene}_{i}.png")
        Image.fromarray(img).save(path)
        print(f"  saved {path}  bright={img.mean():.0f}")
    r.close()


def render_scene_wide(scene, view, w=2520, h=1080, name=None):
    """Render ONE view of a scene at a wide (21:9) aspect for a full-bleed hero
    background. ``view`` = (eye xyz, look xyz) relative to the GLB origin."""
    s = SCENES[scene]
    xform = [float(x) for x in s["xform"]]
    cx, cy = xform[12], xform[13]
    m = mujoco.MjModel.from_xml_string(EMPTY)
    cfg = RendererConfig()
    cfg.width, cfg.height = w, h
    cfg.batch_size = 1
    cfg.enable_shadows = True
    cfg.enable_ssao = True
    cfg.enable_bloom = True
    cfg.exposure = float(s.get("exposure", 0.6))
    r = WarpRenderer(cfg)
    r.load_model(m)
    ii, ss = ibl_paths(s.get("ibl", "studio_ibl"))
    if os.path.exists(ii):
        r.load_ibl(ii, ss)
    r.set_ambient_intensity(float(s.get("ambient", 8000.0)) * 1.5)
    r.load_glb_xform(s["glb"], xform)
    eye, look = view
    r.set_free_camera(cx + eye[0], cy + eye[1], eye[2],
                      cx + look[0], cy + look[1], look[2])
    img = r.render()[..., :3].clamp(0, 255).byte().cpu().numpy()
    path = os.path.join(OUT, f"{name or scene}_wide.png")
    Image.fromarray(img).save(path)
    print(f"  saved {path}  ({w}x{h}, bright={img.mean():.0f})")
    r.close()


def render_warehouse(w=1280, h=800):
    import json
    m = mujoco.MjModel.from_xml_string(EMPTY)
    n = len(WAREHOUSE_VIEWS)
    cfg = RendererConfig()
    cfg.width, cfg.height = w, h
    cfg.batch_size = n
    cfg.enable_shadows = True
    cfg.enable_ssao = True
    cfg.enable_bloom = True
    cfg.exposure = 1.6
    r = WarpRenderer(cfg)
    r.load_model(m)
    ibl = os.path.join(WAREHOUSE_ASSETS, "ibl")
    r.load_ibl(os.path.join(ibl, "warehouse_ibl_ibl.ktx"),
               os.path.join(ibl, "warehouse_ibl_skybox.ktx"))
    r.load_glb(os.path.join(WAREHOUSE_ASSETS, "nvidia_floor.glb"))
    r.load_glb(os.path.join(WAREHOUSE_ASSETS, "nvidia_warehouse_tinted.glb"))
    r.load_glb(os.path.join(WAREHOUSE_ASSETS, "wall_patch.glb"))
    r.set_ambient_intensity(6500.0)
    with open(WAREHOUSE_LAMPS) as f:
        lamps = json.load(f)
    for x, y, z in lamps:
        r.add_spot_light(x, y, z - 0.75, 0.0, 0.0, -1.0,
                         1.0, 0.97, 0.90, 24000000.0, 14.0, 55.0, 88.0)
        r.add_point_light(x, y, z, 1.0, 0.95, 0.85, 90000.0, 2.2)
    for i, (ex, ey, ez, tx, ty, tz) in enumerate(WAREHOUSE_VIEWS):
        r.set_free_camera(ex, ey, ez, tx, ty, tz)
        img = r.render()[..., :3].clamp(0, 255).byte().cpu().numpy()
        path = os.path.join(OUT, f"warehouse_{i}.png")
        Image.fromarray(img).save(path)
        print(f"  saved {path}  bright={img.mean():.0f}")
    r.close()


def main():
    args = sys.argv[1:]
    if args and args[0] == "wide":
        # Wide 21:9 Sponza hero background.
        render_scene_wide("sponza", ((-10.5, 0.0, 2.2), (6.0, 0.2, 1.5)),
                          name="sponza_hero")
        return
    scenes = args or (list(VIEWS) + ["warehouse"])
    for sc in scenes:
        print(f"### {sc} ###")
        try:
            if sc == "warehouse":
                render_warehouse()
            else:
                render_scene(sc)
        except Exception as e:  # keep going; report which scene failed
            print(f"  FAILED {sc}: {e}")


if __name__ == "__main__":
    main()
