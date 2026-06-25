"""M2 validation: layered (single-draw) EGOCENTRIC must match the render_batch
(multi-RT) egocentric ground truth. Same native scene, same per-world robot-
mounted camera. Renders both, reports per-tile mean abs diff + a montage.
"""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)
import numpy as np, torch, mujoco
from mujofil import WarpRenderer, RendererConfig
from PIL import Image

SCENE = """
<mujoco>
  <asset>
    <material name="red"   rgba="0.85 0.2 0.2 1" roughness="0.5"/>
    <material name="green" rgba="0.2 0.8 0.3 1" roughness="0.5"/>
    <material name="blue"  rgba="0.2 0.4 0.9 1" roughness="0.5"/>
    <material name="yellow" rgba="0.9 0.8 0.2 1" roughness="0.5"/>
    <material name="floor" rgba="0.5 0.5 0.55 1" roughness="0.8"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="12 12 0.1" material="floor"/>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="px" type="box" pos="3 0 1" size="0.3 0.3 1" material="red"/>
    <geom name="nx" type="box" pos="-3 0 1" size="0.3 0.3 1" material="green"/>
    <geom name="py" type="box" pos="0 3 1" size="0.3 0.3 1" material="blue"/>
    <geom name="ny" type="box" pos="0 -3 1" size="0.3 0.3 1" material="yellow"/>
    <body name="robot" pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.2" rgba="0.9 0.9 0.95 1"/>
      <camera name="ego" pos="0 0 0.3" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody>
</mujoco>"""


def build(n):
    m = mujoco.MjModel.from_xml_string(SCENE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    datas = []
    for i in range(n):
        d = mujoco.MjData(m)
        yaw = 2 * np.pi * i / n
        adr = m.jnt_qposadr[0]
        d.qpos[adr:adr+3] = [0, 0, 0.5]
        d.qpos[adr+3:adr+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        mujoco.mj_forward(m, d)
        datas.append(d)
    return m, cam, datas


def montage(a, res, path):
    n = a.shape[0]; cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols)); pad = 3
    g = np.zeros((rows*res+(rows-1)*pad, cols*res+(cols-1)*pad, 3), "uint8")
    for i in range(n):
        r, c = divmod(i, cols)
        g[r*(res+pad):r*(res+pad)+res, c*(res+pad):c*(res+pad)+res] = a[i]
    Image.fromarray(g).save(path)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    m, cam, datas = build(n)
    out = os.path.join(HERE, "out", "ego_compare"); os.makedirs(out, exist_ok=True)

    # ground truth: multi-RT per-world camera
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n; cfg.enable_shadows = True
    r = WarpRenderer(cfg); r.load_model(m)
    gt = r.render_batch(m, datas, cam_id=cam)[..., :3].clamp(0, 255).byte().cpu().numpy()
    r.close()

    # layered single-draw per-world camera (the M2 feature)
    cfg2 = RendererConfig(); cfg2.width = cfg2.height = res; cfg2.batch_size = n
    cfg2.layered = True; cfg2.enable_shadows = True
    r2 = WarpRenderer(cfg2); r2.load_model(m)
    ly = r2.render_batch_layered(m, datas, cam_id=cam)[..., :3].clamp(0, 255).byte().cpu().numpy()
    r2.close()

    montage(gt, res, os.path.join(out, f"gt_N{n}.png"))
    montage(ly, res, os.path.join(out, f"layered_N{n}.png"))
    # side-by-side diff per tile
    d = np.abs(gt.astype("int16") - ly.astype("int16"))
    per_tile = d.reshape(n, -1).mean(1)
    uniq_ly = len(set(ly[i].tobytes() for i in range(n)))
    print(f"N={n} res={res}")
    print(f"  layered uniq tiles = {uniq_ly}/{n}")
    print(f"  per-tile mean|diff| vs ground truth = {[round(float(x),1) for x in per_tile]}")
    print(f"  overall mean|diff| = {float(d.mean()):.2f}  max = {int(d.max())}")
    print(f"  saved {out}/gt_N{n}.png and layered_N{n}.png")


if __name__ == "__main__":
    main()
