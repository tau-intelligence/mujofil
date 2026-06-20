"""Warehouse LAYERED render: N worlds of the photoreal warehouse + per-world
objects in ONE instanced GPU pass. Saves a montage to eyeball whether the GLB
backdrop appears in every layer, and reports throughput.

  MUJOFIL_WARP_BACKEND=gl python test_layered_warehouse.py [N]
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujofil
_warp_mats = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujofil_warp", "materials")
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", _warp_mats)

import numpy as np
import torch
import mujoco
from mujofil_warp import WarpRenderer, RendererConfig

WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
A = os.path.join(WH, "assets")

SCENE = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.06"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.20"/>
    <material name="copper" rgba="0.95 0.6 0.45 1"  metallic="1.0" roughness="0.30"/>
    <material name="red"    rgba="0.80 0.18 0.15 1" metallic="0.0" roughness="0.55"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.40"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    <body pos="-0.6 0 0.45"><freejoint/><geom type="sphere" size="0.30" material="chrome"/></body>
    <body pos="0.1 -0.3 0.30"><freejoint/><geom type="box" size="0.24 0.24 0.24" material="gold"/></body>
    <body pos="0.6 0.3 0.26"><freejoint/><geom type="sphere" size="0.22" material="copper"/></body>
    <body pos="0.0 0.6 0.20"><freejoint/><geom type="capsule" size="0.12 0.18" material="red"/></body>
    <camera name="cam0" pos="0 -3.0 1.2" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>
"""


def _warehouse(r):
    r.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
               os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        r.load_glb(os.path.join(A, g))
    r.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        r.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    res = 256
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(N)]
    rng = np.random.default_rng(0)
    for i, d in enumerate(datas):
        # drop objects, settle, distinct per-world spread
        for b in range(m.njnt):
            d.qpos[7 * b + 0] += 0.25 * ((i % 5) - 2) + rng.uniform(-0.05, 0.05)
            d.qpos[7 * b + 2] += 0.4
        mujoco.mj_forward(m, d)
        for _ in range(300):
            mujoco.mj_step(m, d)

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = N
    cfg.layered = True
    cfg.exposure = 1.6
    r = WarpRenderer(cfg)
    r.load_model(m)
    _warehouse(r)
    print("geom_count =", r.geom_count, " layered =", r.layered)

    # warmup + time
    for _ in range(3):
        imgs = r.render_batch_layered(m, datas, cam_id=0)
    torch.cuda.synchronize()
    r.reset_profile()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        imgs = r.render_batch_layered(m, datas, cam_id=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p = r.profile()
    print(f"LAYERED warehouse: {N*iters/dt:.0f} cam/s  "
          f"render={p['render_ms']/iters:.2f} flush={p['flush_ms']/iters:.2f} copy={p['copy_ms']/iters:.2f} ms")

    # montage to eyeball backdrop-in-every-layer
    from PIL import Image
    os.makedirs("out", exist_ok=True)
    cell = imgs[..., :3].cpu().numpy()[:, ::-1]
    cols = min(N, 4); rows = (N + cols - 1) // cols
    canvas = np.zeros((rows * res, cols * res, 3), np.uint8)
    means = []
    for i in range(N):
        rr, cc = divmod(i, cols)
        canvas[rr*res:(rr+1)*res, cc*res:(cc+1)*res] = cell[i]
        means.append(round(float(cell[i].mean()), 1))
    Image.fromarray(canvas).save("out/layered_warehouse.png")
    print("per-world mean brightness:", means)
    print("saved out/layered_warehouse.png")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
