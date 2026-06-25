"""Show the full-quality render_batch egocentric output (the path we should SHIP)
next to the washed-out layered, same scene + same robot cameras, as a montage.
"""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, mujoco, torch
from mujofil import WarpRenderer, RendererConfig
from trailer.scenes import SCENES
from PIL import Image

ROBOT = """<mujoco><worldbody><body name="robot" pos="0 0 0.4"><freejoint/>
<geom type="box" size="0.2 0.15 0.1" rgba="0 0 0 0"/>
<camera name="ego" pos="0.2 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/></body></worldbody></mujoco>"""
d_ibl = os.path.join(HERE, "assets", "ibl", "studio")
IBL = (os.path.join(d_ibl, "studio_ibl_ibl.ktx"), os.path.join(d_ibl, "studio_ibl_skybox.ktx"))


def montage(a, res, pad=4, bg=20):
    n = a.shape[0]; cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols))
    g = np.full((rows*res+(rows+1)*pad, cols*res+(cols+1)*pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols); y = pad+r*(res+pad); x = pad+c*(res+pad)
        g[y:y+res, x:x+res] = a[i]
    return g


def datas(m, n, cx, cy):
    out = []
    for i in range(n):
        d = mujoco.MjData(m); a = 2*np.pi*i/n
        j = m.jnt_qposadr[0]
        d.qpos[j:j+3] = [cx+1.4*np.cos(a), cy+1.4*np.sin(a), 0.4]
        h = a+np.pi/2; d.qpos[j+3:j+7] = [np.cos(h/2), 0, 0, np.sin(h/2)]
        mujoco.mj_forward(m, d); out.append(d)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "sponza"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 320
    s = SCENES[scene]; xf = [float(x) for x in s["xform"]]; cx, cy = xf[12], xf[13]
    m = mujoco.MjModel.from_xml_string(ROBOT)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    ds = datas(m, n, cx, cy)

    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n; cfg.enable_shadows = True
    r = WarpRenderer(cfg); r.load_model(m); r.load_ibl(*IBL)
    r.set_ambient_intensity(float(s.get("ambient", 8000))*1.5)
    r.load_glb_xform(s["glb"], xf)
    img = r.render_batch(m, ds, cam_id=cam)[..., :3].clamp(0, 255).byte().cpu().numpy()
    r.close()

    out = os.path.join(HERE, "out", "batch_quality"); os.makedirs(out, exist_ok=True)
    Image.fromarray(montage(img, res)).save(os.path.join(out, f"render_batch_{scene}_N{n}.png"))
    print(f"saved {out}/render_batch_{scene}_N{n}.png  (full-quality render_batch egocentric, bright={img.mean():.0f})")


if __name__ == "__main__":
    main()
