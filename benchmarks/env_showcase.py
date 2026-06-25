"""Clean, well-framed stills of the photoreal GLB environments (living + cafe),
rendered with the high-fidelity render_batch path (full Filament PBR + IBL + in-
engine tonemap). Eye-level cameras placed inside each room looking across it.
"""
import os, sys
os.environ.setdefault("MUJOFIL_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, torch, mujoco
from mujofil import WarpRenderer, RendererConfig
from PIL import Image

EMPTY = "<mujoco><worldbody/></mujoco>"   # no robot; just the environment


def ibl():
    d = os.path.join(HERE, "assets", "ibl", "studio")
    return os.path.join(d, "studio_ibl_ibl.ktx"), os.path.join(d, "studio_ibl_skybox.ktx")


# A few hand-framed eye-level viewpoints per scene (eye xyz -> look xyz), chosen to
# look ACROSS the room interior. cx,cy is the GLB origin (xform translation).
VIEWS = {
    "living": [
        ((-3.0, 2.4, 1.4), (0.5, -0.5, 0.7)),
        (( 2.6, 2.6, 1.5), (-1.0, -0.8, 0.6)),
        ((-2.4, -2.2, 1.4), (1.0, 1.0, 0.7)),
        (( 0.2, 3.0, 1.6), (0.0, -2.0, 0.6)),
    ],
    "cafe": [
        ((-3.0, 2.8, 1.5), (0.6, -0.8, 0.8)),
        (( 2.8, 2.4, 1.5), (-1.0, -0.8, 0.8)),
        ((-2.6, -2.4, 1.5), (1.0, 1.0, 0.9)),
        (( 0.0, 3.2, 1.6), (0.0, -2.0, 0.8)),
    ],
}


def montage(a, res, pad=6, bg=24):
    n = a.shape[0]; cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols))
    g = np.full((rows*res+(rows+1)*pad, cols*res+(cols+1)*pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols); y = pad+r*(res+pad); x = pad+c*(res+pad)
        g[y:y+res, x:x+res] = a[i]
    return g


def render_scene(scene, res=480):
    from trailer.scenes import SCENES
    s = SCENES[scene]; xform = [float(x) for x in s["xform"]]
    cx, cy = xform[12], xform[13]
    views = VIEWS[scene]
    n = len(views)
    m = mujoco.MjModel.from_xml_string(EMPTY)
    datas = [mujoco.MjData(m) for _ in range(n)]   # identical empty worlds

    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_shadows = True; cfg.exposure = 1.0
    r = WarpRenderer(cfg); r.load_model(m)
    ii, ss = ibl()
    if os.path.exists(ii): r.load_ibl(ii, ss)
    r.set_ambient_intensity(float(s.get("ambient", 8000.0)) * 1.5)
    r.load_glb_xform(s["glb"], xform)

    # Render each view with the free camera (one render() per view, batched sync).
    imgs = []
    for (eye, look) in views:
        r.set_free_camera(cx+eye[0], cy+eye[1], eye[2], cx+look[0], cy+look[1], look[2])
        img = r.render()[..., :3].clamp(0, 255).byte().cpu().numpy()
        imgs.append(img)
    a = np.stack(imgs)
    out = os.path.join(HERE, "out", "env_renders"); os.makedirs(out, exist_ok=True)
    Image.fromarray(montage(a, res)).save(os.path.join(out, f"{scene}_views.png"))
    print(f"saved {out}/{scene}_views.png  ({n} views, {res}px, bright={a.mean():.0f})")
    r.close()


def main():
    for scene in (sys.argv[1:] or ["living", "cafe"]):
        render_scene(scene)


if __name__ == "__main__":
    main()
