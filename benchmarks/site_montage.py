"""Parallel-batch montage for the site: N distinct MuJoCo worlds rendered in ONE
instanced layered GPU pass, tiled into a grid -- the visual proof of the
parallel renderer. Each world has differently-posed PBR objects.
"""
import os
import sys

os.environ.setdefault("MUJOFIL_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR",
                      os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)

import numpy as np
import mujoco
from PIL import Image
from mujofil import WarpRenderer, RendererConfig

OUT = os.path.join(HERE, "site_assets", "img")
os.makedirs(OUT, exist_ok=True)

SCENE = """
<mujoco>
  <option timestep="0.004"/>
  <visual><global offwidth="512" offheight="512"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.07"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.20"/>
    <material name="copper" rgba="0.95 0.55 0.35 1" metallic="1.0" roughness="0.30"/>
    <material name="teal"   rgba="0.20 0.62 0.70 1" metallic="0.3" roughness="0.35"/>
    <material name="floor"  rgba="0.20 0.19 0.18 1" metallic="0.0" roughness="0.45"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"/>
    <body pos="-0.6 0 0.9"><freejoint/><geom type="sphere" size="0.25" material="chrome"/></body>
    <body pos="0.1 0 1.1"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
    <body pos="0.8 0.3 0.8"><freejoint/><geom type="sphere" size="0.22" material="copper"/></body>
    <body pos="-0.2 0.7 1.0"><freejoint/><geom type="capsule" size="0.16 0.2" material="teal"/></body>
    <camera name="cam0" pos="0 -4.2 2.0" xyaxes="1 0 0 0 0.5 0.86"/>
  </worldbody>
</mujoco>
"""


def montage(a, pad=4, bg=20):
    n, h, w, _ = a.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    g = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        g[y:y + h, x:x + w] = a[i]
    return g


def main(n=16, res=320):
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    # Settle each world a different number of steps so the falling objects land
    # in visibly distinct poses across the batch.
    for i, d in enumerate(datas):
        for _ in range(14 + i * 2):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    cfg.enable_shadows = True
    cfg.enable_ssao = True
    cfg.enable_bloom = True
    cfg.exposure = 0.2
    r = WarpRenderer(cfg)
    r.load_model(m)
    ibl = os.path.join(HERE, "assets", "ibl", "studio")
    ii = os.path.join(ibl, "studio_ibl_ibl.ktx")
    ss = os.path.join(ibl, "studio_ibl_skybox.ktx")
    if os.path.exists(ii):
        r.load_ibl(ii, ss)
    r.set_ambient_intensity(9000.0)
    a = r.render_batch(m, datas, cam_id=0)[..., :3].clamp(0, 255).byte().cpu().numpy()
    Image.fromarray(montage(a)).save(os.path.join(OUT, "parallel_montage.png"))
    print(f"saved {OUT}/parallel_montage.png  ({n} worlds, {res}px, bright={a.mean():.0f})")
    r.close()


if __name__ == "__main__":
    main()
