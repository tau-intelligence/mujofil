"""Verify the shippable public API: WarpRenderer(preset="eval") vs preset="train"
(now SSR-off) on egocentric photoreal GLB. train must be ~4-5x faster AND stay
photoreal (full gltfio PBR + FILMIC). Saves a train-preset montage for the eye.
"""
import os, sys, time
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil_warp", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, mujoco, torch
from mujofil_warp import WarpRenderer
from trailer.scenes import SCENES
from PIL import Image

ROBOT = """<mujoco><worldbody><body name="robot" pos="0 0 0.4"><freejoint/>
<geom type="box" size="0.2 0.15 0.1" rgba="0 0 0 0"/>
<camera name="ego" pos="0.2 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/></body></worldbody></mujoco>"""
d_ibl = os.path.join(HERE, "assets", "ibl", "studio")
IBL = (os.path.join(d_ibl, "studio_ibl_ibl.ktx"), os.path.join(d_ibl, "studio_ibl_skybox.ktx"))


def datas(m, n, cx, cy):
    out = []
    for i in range(n):
        d = mujoco.MjData(m); a = 2*np.pi*i/n; j = m.jnt_qposadr[0]
        d.qpos[j:j+3] = [cx+1.4*np.cos(a), cy+1.4*np.sin(a), 0.4]
        h = a+np.pi/2; d.qpos[j+3:j+7] = [np.cos(h/2), 0, 0, np.sin(h/2)]
        mujoco.mj_forward(m, d); out.append(d)
    return out


def montage(a, res, pad=4, bg=20):
    n = a.shape[0]; cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols))
    g = np.full((rows*res+(rows+1)*pad, cols*res+(cols+1)*pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols); g[pad+r*(res+pad):pad+r*(res+pad)+res, pad+c*(res+pad):pad+c*(res+pad)+res] = a[i]
    return g


def run(scene, res, n, preset, save=False):
    s = SCENES[scene]; xf = [float(x) for x in s["xform"]]; cx, cy = xf[12], xf[13]
    m = mujoco.MjModel.from_xml_string(ROBOT)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    ds = datas(m, n, cx, cy)
    r = WarpRenderer(width=res, height=res, batch_size=n, preset=preset)
    r.load_model(m); r.load_ibl(*IBL)
    r.set_ambient_intensity(float(s.get("ambient", 8000))*1.5); r.load_glb_xform(s["glb"], xf)

    def step(): return r.render_batch(m, ds, cam_id=cam)
    for _ in range(3): step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(8): step()
    torch.cuda.synchronize(); cps = n*8/(time.perf_counter()-t0)
    if save:
        img = step()[..., :3].clamp(0, 255).byte().cpu().numpy()
        out = os.path.join(HERE, "out", "preset_check"); os.makedirs(out, exist_ok=True)
        Image.fromarray(montage(img, res)).save(os.path.join(out, f"{scene}_{preset}_N{n}.png"))
    r.close()
    return cps


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    scenes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["cafe", "sponza"]
    print(f"public preset API, render_batch egocentric, res={res} N={n}\n")
    print(f"{'scene':>8} | {'eval cam/s':>10} | {'train cam/s':>11} | {'speedup':>8}")
    for scene in scenes:
        ev = run(scene, res, n, "eval")
        tr = run(scene, res, n, "train", save=True)
        print(f"{scene:>8} | {ev:>10.0f} | {tr:>11.0f} | {tr/ev:>7.2f}x")
    print(f"\nsaved train-preset montages in out/preset_check/")


if __name__ == "__main__":
    main()
