"""Reload/leak stress: repeatedly load a GLB-ingest layered scene, render, reload,
close -- in one process, watching GPU memory. Catches texture/instance leaks on
the GLB-ingest + float16 path (sponza has 100s of textures, the worst case).
"""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, mujoco, torch
from mujofil import WarpRenderer, RendererConfig
from trailer.scenes import SCENES

ROBOT = """<mujoco><worldbody><body name="robot" pos="0 0 0.4"><freejoint/>
<geom type="box" size="0.2 0.15 0.1" rgba="0 0 0 0"/>
<camera name="ego" pos="0.2 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/></body></worldbody></mujoco>"""
d_ibl = os.path.join(HERE, "assets", "ibl", "studio")
IBL = (os.path.join(d_ibl, "studio_ibl_ibl.ktx"), os.path.join(d_ibl, "studio_ibl_skybox.ktx"))


def free_mb():
    free, total = torch.cuda.mem_get_info()
    return free / 1e6


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "cafe"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    s = SCENES[scene]; xf = [float(x) for x in s["xform"]]; cx, cy = xf[12], xf[13]
    m = mujoco.MjModel.from_xml_string(ROBOT)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    ds = []
    for i in range(4):
        d = mujoco.MjData(m); a = 2*np.pi*i/4; j = m.jnt_qposadr[0]
        d.qpos[j:j+3] = [cx+1.4*np.cos(a), cy+1.4*np.sin(a), 0.4]
        mujoco.mj_forward(m, d); ds.append(d)

    torch.cuda.synchronize()
    base = free_mb()
    print(f"baseline free = {base:.0f} MB")
    for k in range(iters):
        r = WarpRenderer(width=128, height=128, batch_size=4, layered=True)
        r.load_model(m); r.load_ibl(*IBL); r.set_ambient_intensity(8000.0)
        r.load_glb_layered(s["glb"], xf)
        r.render_batch_layered(m, ds, cam_id=cam)
        r.load_model(m)  # reload same renderer (exercises clear())
        r.load_glb_layered(s["glb"], xf)
        r.render_batch_layered(m, ds, cam_id=-1)
        r.close()
        torch.cuda.synchronize()
        print(f"  iter {k+1}: free = {free_mb():.0f} MB  (delta vs base {free_mb()-base:+.0f})")
    leak = base - free_mb()
    print(f"\nnet leak after {iters} create/reload/close cycles = {leak:.0f} MB")
    if leak > 200:
        print("LEAK SUSPECTED (>200MB)"); sys.exit(1)
    print("OK: no significant leak")


if __name__ == "__main__":
    main()
