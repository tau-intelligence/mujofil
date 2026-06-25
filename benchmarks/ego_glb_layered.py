"""Validate + benchmark EGOCENTRIC over a photoreal GLB in the LAYERED single draw,
via load_glb_layered (GLB ingested as instanced renderables). Compares vs the
render_batch ground truth (same scene, same robot cameras) and times both.
"""
import os, sys, time
os.environ.setdefault("MUJOFIL_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, torch, mujoco
from mujofil import WarpRenderer, RendererConfig
from PIL import Image

ROBOT = """
<mujoco><option timestep="0.01"/><worldbody>
  <body name="robot" pos="0 0 0.4"><freejoint/>
    <geom type="box" size="0.22 0.16 0.10" rgba="0.12 0.12 0.14 1"/>
    <camera name="ego" pos="0.22 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/>
  </body></worldbody></mujoco>"""


def ibl():
    d = os.path.join(HERE, "assets", "ibl", "studio")
    return os.path.join(d, "studio_ibl_ibl.ktx"), os.path.join(d, "studio_ibl_skybox.ktx")


def montage(a, res, pad=4, bg=16):
    n = a.shape[0]; cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols))
    g = np.full((rows*res+(rows+1)*pad, cols*res+(cols+1)*pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols); y = pad+r*(res+pad); x = pad+c*(res+pad)
        g[y:y+res, x:x+res] = a[i]
    return g


def datas(m, n, cx, cy):
    jadr = m.jnt_qposadr[0]; out = []
    for i in range(n):
        d = mujoco.MjData(m); aa = 2*np.pi*i/n
        d.qpos[jadr:jadr+3] = [cx+1.4*np.cos(aa), cy+1.4*np.sin(aa), 0.4]
        h = aa+np.pi/2; d.qpos[jadr+3:jadr+7] = [np.cos(h/2), 0, 0, np.sin(h/2)]
        mujoco.mj_forward(m, d); out.append(d)
    return out


def timed(fn, iters=6, warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "living"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    from trailer.scenes import SCENES
    s = SCENES[scene]; xform = [float(x) for x in s["xform"]]
    glb = s["glb"]; ambient = float(s.get("ambient", 8000.0))
    cx, cy = xform[12], xform[13]
    ii, ss = ibl()
    m = mujoco.MjModel.from_xml_string(ROBOT)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    ds = datas(m, n, cx, cy)
    out = os.path.join(HERE, "out", "ego_glb_layered"); os.makedirs(out, exist_ok=True)

    # ---- A) NEW: layered single-draw egocentric over the ingested GLB ----
    cfgL = RendererConfig(); cfgL.width = cfgL.height = res; cfgL.batch_size = n
    cfgL.layered = True; cfgL.enable_shadows = True; cfgL.exposure = 1.0
    rL = WarpRenderer(cfgL); rL.load_model(m)
    if os.path.exists(ii): rL.load_ibl(ii, ss)
    # The layered OBJECTS pass has post-processing OFF, so it outputs LINEAR colour
    # which the Python ACES tonemap brings to display range. Light it with IBL
    # ambient ONLY (soft, even) -- NO harsh directional fills, which would add a
    # broad specular sheen that reads as a fake "metallic" finish.
    rL.set_ambient_intensity(max(ambient * 2.0, 16000.0))
    t0 = time.perf_counter()
    nmesh = rL.load_glb_layered(glb, xform)
    print(f"ingested {nmesh} GLB meshes in {time.perf_counter()-t0:.1f}s")
    # DETERMINISTIC auto-exposure (exposure=None) -- no per-scene magic number.
    lay = rL.render_batch_layered(m, ds, cam_id=cam)[..., :3].clamp(0, 255).byte().cpu().numpy()
    tL = timed(lambda: rL.render_batch_layered(m, ds, cam_id=cam))
    rL.close()

    # ---- B) ground truth: render_batch egocentric over the gltfio GLB ----
    cfgB = RendererConfig(); cfgB.width = cfgB.height = res; cfgB.batch_size = n
    cfgB.enable_shadows = True; cfgB.exposure = 1.0
    rB = WarpRenderer(cfgB); rB.load_model(m)
    if os.path.exists(ii): rB.load_ibl(ii, ss)
    rB.set_ambient_intensity(ambient)
    rB.load_glb_xform(glb, xform)
    gt = rB.render_batch(m, ds, cam_id=cam)[..., :3].clamp(0, 255).byte().cpu().numpy()
    tB = timed(lambda: rB.render_batch(m, ds, cam_id=cam))
    rB.close()

    Image.fromarray(montage(lay, res)).save(os.path.join(out, f"layered_{scene}_N{n}.png"))
    Image.fromarray(montage(gt, res)).save(os.path.join(out, f"groundtruth_{scene}_N{n}.png"))
    uniq = len(set(lay[i].tobytes() for i in range(n)))
    nonblack = int((lay.reshape(n, -1).mean(1) > 8).sum())
    print(f"\nscene={scene} N={n} res={res}")
    print(f"  layered ingest: uniq={uniq}/{n}  nonblack={nonblack}/{n}  mean_bright={lay.mean():.1f}")
    print(f"  PERF  layered single-draw: {n/tL:8.0f} cam/s ({tL*1000:.1f} ms)")
    print(f"  PERF  render_batch (GT)   : {n/tB:8.0f} cam/s ({tB*1000:.1f} ms)")
    print(f"  speedup layered/render_batch = {(n/tL)/(n/tB):.1f}x")
    print(f"  saved {out}/layered_{scene}_N{n}.png  &  groundtruth_{scene}_N{n}.png")


if __name__ == "__main__":
    main()
