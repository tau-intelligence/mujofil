"""Scaling: egocentric photoreal GLB, layered single-draw (load_glb_layered) vs
render_batch, across N. The instanced ingest should widen the speedup with N the
same way native layered does, because it collapses N egocentric env views into
ONE draw.
"""
import os, sys, time
os.environ.setdefault("MUJOFIL_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, torch, mujoco
from mujofil import WarpRenderer, RendererConfig

ROBOT = """
<mujoco><option timestep="0.01"/><worldbody>
  <body name="robot" pos="0 0 0.4"><freejoint/>
    <geom type="box" size="0.22 0.16 0.10" rgba="0.12 0.12 0.14 1"/>
    <camera name="ego" pos="0.22 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/>
  </body></worldbody></mujoco>"""


def ibl():
    d = os.path.join(HERE, "assets", "ibl", "studio")
    return os.path.join(d, "studio_ibl_ibl.ktx"), os.path.join(d, "studio_ibl_skybox.ktx")


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
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    Ns = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["16","64","256"])]
    from trailer.scenes import SCENES
    s = SCENES[scene]; xform = [float(x) for x in s["xform"]]; glb = s["glb"]
    ambient = float(s.get("ambient", 8000.0)); cx, cy = xform[12], xform[13]
    ii, ss = ibl()
    m = mujoco.MjModel.from_xml_string(ROBOT)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    print(f"EGOCENTRIC photoreal GLB '{scene}', res={res}  (cameras/sec)\n")
    print(f"{'N':>5} | {'layered (1 draw)':>16} | {'render_batch':>13} | {'speedup':>8}")
    for n in Ns:
        ds = datas(m, n, cx, cy)
        cfgL = RendererConfig(); cfgL.width = cfgL.height = res; cfgL.batch_size = n
        cfgL.layered = True; cfgL.enable_shadows = True
        rL = WarpRenderer(cfgL); rL.load_model(m)
        if os.path.exists(ii): rL.load_ibl(ii, ss)
        rL.set_ambient_intensity(ambient); rL.load_glb_layered(glb, xform)
        tL = timed(lambda: rL.render_batch_layered(m, ds, cam_id=cam)); rL.close()

        cfgB = RendererConfig(); cfgB.width = cfgB.height = res; cfgB.batch_size = n
        cfgB.enable_shadows = True
        rB = WarpRenderer(cfgB); rB.load_model(m)
        if os.path.exists(ii): rB.load_ibl(ii, ss)
        rB.set_ambient_intensity(ambient); rB.load_glb_xform(glb, xform)
        tB = timed(lambda: rB.render_batch(m, ds, cam_id=cam)); rB.close()
        print(f"{n:>5} | {n/tL:>16.0f} | {n/tB:>13.0f} | {(n/tL)/(n/tB):>7.1f}x")


if __name__ == "__main__":
    main()
