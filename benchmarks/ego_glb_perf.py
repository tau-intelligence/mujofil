"""Measure egocentric throughput on a photoreal GLB scene, three ways, to make the
'is it still 19x?' tradeoff concrete (cameras/sec, higher = better):

  A) render_batch  EGOCENTRIC (cam per world)  -- what egocentric GLB actually uses
  B) render_batch  SHARED     (one camera)     -- same path, shared cam baseline
  C) render_batch_layered SHARED               -- the single-draw path (the ~19x trick)

A vs C shows what egocentric "costs" vs the fast shared-camera layered path. A vs B
shows that egocentric is ~free relative to render_batch itself (same path).
"""
import os, sys, time
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil_warp", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, torch, mujoco
from mujofil_warp import WarpRenderer, RendererConfig

ROBOT = """
<mujoco><option timestep="0.01"/><worldbody>
  <body name="robot" pos="0 0 0.4"><freejoint/>
    <geom type="box" size="0.22 0.16 0.10" rgba="0.12 0.12 0.14 1"/>
    <camera name="ego" pos="0.22 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/>
  </body></worldbody></mujoco>"""


def ibl_paths(name):
    base = os.path.join(HERE, "assets", "ibl")
    for cand in (name, name.replace("_ibl", ""), "studio"):
        d = os.path.join(base, cand)
        ii = os.path.join(d, f"{cand}_ibl.ktx"); ss = os.path.join(d, f"{cand}_skybox.ktx")
        if os.path.exists(ii):
            return ii, ss
    d = os.path.join(base, "studio")
    return os.path.join(d, "studio_ibl_ibl.ktx"), os.path.join(d, "studio_ibl_skybox.ktx")


def build(scene, n, res, layered):
    from trailer.scenes import SCENES
    s = SCENES[scene]
    xform = [float(x) for x in s["xform"]]
    m = mujoco.MjModel.from_xml_string(ROBOT)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    jadr = m.jnt_qposadr[0]
    cx, cy = xform[12], xform[13]
    datas = []
    for i in range(n):
        d = mujoco.MjData(m)
        a = 2 * np.pi * i / n
        d.qpos[jadr:jadr+3] = [cx + 1.4*np.cos(a), cy + 1.4*np.sin(a), 0.4]
        h = a + np.pi/2
        d.qpos[jadr+3:jadr+7] = [np.cos(h/2), 0, 0, np.sin(h/2)]
        mujoco.mj_forward(m, d)
        datas.append(d)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_shadows = True; cfg.layered = layered
    r = WarpRenderer(cfg); r.load_model(m)
    ii, ss = ibl_paths(s.get("ibl", "studio_ibl"))
    if os.path.exists(ii):
        r.load_ibl(ii, ss)
    r.set_ambient_intensity(float(s.get("ambient", 8000.0)))
    r.load_glb_xform(s["glb"], xform)
    return m, datas, cam, r


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "cafe"
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    Ns = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["16", "64"])]
    print(f"scene={scene} res={res}  (cameras/sec, higher=better)")
    print(f"{'N':>5} | {'A render_batch EGO':>20} | {'B render_batch SHARED':>22} | {'C layered SHARED':>18} | {'C/A speedup':>11}")
    for n in Ns:
        m, datas, cam, r = build(scene, n, res, layered=False)
        a = timed(lambda: r.render_batch(m, datas, cam_id=cam))
        b = timed(lambda: r.render_batch(m, datas, cam_id=-1))
        r.close()
        m2, d2, cam2, r2 = build(scene, n, res, layered=True)
        c = timed(lambda: r2.render_batch_layered(m2, d2, cam_id=-1))
        r2.close()
        cps_a, cps_b, cps_c = n/a, n/b, n/c
        print(f"{n:>5} | {cps_a:>20.0f} | {cps_b:>22.0f} | {cps_c:>18.0f} | {cps_c/cps_a:>10.1f}x")


if __name__ == "__main__":
    main()
