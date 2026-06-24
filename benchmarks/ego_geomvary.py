"""Diagnose M2 geometry-collapse: FIXED camera, per-world DIFFERENT box position.
Compare render_batch (multi-RT ground truth) vs render_batch_layered egocentric.
Saves both montages so we can SEE which geometry renders."""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil_warp", "materials"))
sys.path.insert(0, HERE)
import numpy as np, torch, mujoco
from mujofil_warp import WarpRenderer, RendererConfig
from PIL import Image

SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
    <light pos="0 0 4"/>
    <body name="b0" pos="0 0 0.4"><freejoint/>
      <geom type="box" size="0.3 0.3 0.3" rgba="0.85 0.25 0.2 1"/></body>
    <camera name="c" pos="0 -2.6 1.0" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>"""


def montage(a, res, path):
    n = a.shape[0]; cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols)); pad = 3
    g = np.zeros((rows*res+(rows-1)*pad, cols*res+(cols-1)*pad, 3), "uint8")
    for i in range(n):
        r, c = divmod(i, cols)
        g[r*(res+pad):r*(res+pad)+res, c*(res+pad):c*(res+pad)+res] = a[i]
    Image.fromarray(g).save(path)


def main():
    n = 6; res = 192
    m = mujoco.MjModel.from_xml_string(SCENE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "c")
    datas = []
    for i in range(n):
        d = mujoco.MjData(m)
        adr = m.jnt_qposadr[0]
        # spread the box across the floor per world (distinct geometry, fixed cam)
        d.qpos[adr:adr+3] = [(i - n/2) * 0.7, 0.0, 0.4]
        mujoco.mj_forward(m, d)
        datas.append(d)
    out = os.path.join(HERE, "out", "ego_geomvary"); os.makedirs(out, exist_ok=True)

    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    r = WarpRenderer(cfg); r.load_model(m)
    gt = r.render_batch(m, datas, cam_id=cam)[..., :3].clamp(0, 255).byte().cpu().numpy()
    r.close()

    cfg2 = RendererConfig(); cfg2.width = cfg2.height = res; cfg2.batch_size = n; cfg2.layered = True
    r2 = WarpRenderer(cfg2); r2.load_model(m)
    ly = r2.render_batch_layered(m, datas, cam_id=cam)[..., :3].clamp(0, 255).byte().cpu().numpy()
    # shared-camera layered (free cam) on the SAME varying geometry -> sanity that
    # the InstanceBuffer per-world transforms still vary post-M2-changes.
    sh = r2.render_batch_layered(m, datas, cam_id=-1)[..., :3].clamp(0, 255).byte().cpu().numpy()
    r2.close()

    montage(gt, res, os.path.join(out, "gt.png"))
    montage(ly, res, os.path.join(out, "layered.png"))
    montage(sh, res, os.path.join(out, "shared.png"))

    def box_centroid(t):
        rr, g, b = t[..., 0].astype("int16"), t[..., 1].astype("int16"), t[..., 2].astype("int16")
        msk = (rr > g + 30) & (rr > b + 30)  # red box
        if msk.sum() < 20: return None
        return round(float(np.where(msk.any(0))[0].mean()), 1)
    print("FIXED camera, per-world DIFFERENT box x-position:")
    print("  GT box centroid_x:", [box_centroid(gt[i]) for i in range(n)])
    print("  LY box centroid_x:", [box_centroid(ly[i]) for i in range(n)])
    print("  SH box centroid_x:", [box_centroid(sh[i]) for i in range(n)])
    print("  layered(ego) uniq:", len(set(ly[i].tobytes() for i in range(n))), "/", n)
    print("  layered(shared) uniq:", len(set(sh[i].tobytes() for i in range(n))), "/", n)
    d = np.abs(gt.astype("int16") - ly.astype("int16"))
    print(f"  mean|diff|={float(d.mean()):.1f} max={int(d.max())}")
    print("  saved", out)


if __name__ == "__main__":
    main()
