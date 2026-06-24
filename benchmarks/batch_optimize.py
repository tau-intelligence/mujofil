"""Optimize the EXISTING full-quality render_batch instead of the washed-out
layered path. Compare, on egocentric photoreal GLB (full Filament PBR + post +
FILMIC, looks like the batch reference):
  A) render_batch  multi-RT  (atlas off, current default)
  B) render_batch  ATLAS     (single-RT, ONE GPU sync, PIXEL-IDENTICAL full quality)
  C) layered ingest          (fast but washed-out -- the thing we're moving away from)
Metric = cameras/sec. A vs B = the free speedup available with ZERO quality loss.
"""
import os, sys, time, subprocess, json
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"


CHILD = r'''
import os, sys, time
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil_warp", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, mujoco, torch
from mujofil_warp import WarpRenderer, RendererConfig
from trailer.scenes import SCENES
ROBOT = """<mujoco><worldbody><body name="robot" pos="0 0 0.4"><freejoint/>
<geom type="box" size="0.2 0.15 0.1" rgba="0 0 0 0"/>
<camera name="ego" pos="0.2 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/></body></worldbody></mujoco>"""
d_ibl = os.path.join(HERE, "assets", "ibl", "studio")
IBL = (os.path.join(d_ibl,"studio_ibl_ibl.ktx"), os.path.join(d_ibl,"studio_ibl_skybox.ktx"))

mode, scene, res, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = SCENES[scene]; xf=[float(x) for x in s["xform"]]; cx,cy=xf[12],xf[13]
m = mujoco.MjModel.from_xml_string(ROBOT)
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
ds=[]
for i in range(n):
    d=mujoco.MjData(m); a=2*np.pi*i/n
    d.qpos[m.jnt_qposadr[0]:m.jnt_qposadr[0]+3]=[cx+1.4*np.cos(a),cy+1.4*np.sin(a),0.4]
    h=a+np.pi/2; d.qpos[m.jnt_qposadr[0]+3:m.jnt_qposadr[0]+7]=[np.cos(h/2),0,0,np.sin(h/2)]
    mujoco.mj_forward(m,d); ds.append(d)

layered = (mode=="layered")
cfg=RendererConfig(); cfg.width=cfg.height=res; cfg.batch_size=n
cfg.enable_shadows=True; cfg.layered=layered
r=WarpRenderer(cfg); r.load_model(m); r.load_ibl(*IBL)
r.set_ambient_intensity(float(s.get("ambient",8000))*1.5)
if layered: r.load_glb_layered(s["glb"], xf)
else:       r.load_glb_xform(s["glb"], xf)

def step():
    if layered: return r.render_batch_layered(m, ds, cam_id=cam)
    return r.render_batch(m, ds, cam_id=cam)
for _ in range(3): step()
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(8): step()
torch.cuda.synchronize()
print("RESULT", n*8/(time.perf_counter()-t0))
'''


def run(mode, scene, res, n, atlas):
    env = "MUJOFIL_WARP_ATLAS=1 " if atlas else ""
    cmd = (f"cd {HERE} && source .venv/bin/activate && {env}"
           f"PYTHONPATH={HERE} MUJOFIL_WARP_BACKEND=gl python -c '{CHILD}' {mode} {scene} {res} {n}")
    full = ["systemd-run","--user","-p","LimitAS=infinity","--quiet","--wait","--pipe","--",
            "bash","-c",cmd]
    try:
        out = subprocess.run(full, capture_output=True, text=True, timeout=600)
        for ln in out.stdout.splitlines():
            if ln.startswith("RESULT"): return float(ln.split()[1])
    except Exception:
        pass
    return None


def main():
    res = int(sys.argv[1]) if len(sys.argv)>1 else 256
    Ns = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["16","64"])]
    scenes = sys.argv[3].split(",") if len(sys.argv)>3 else ["cafe","sponza"]
    print(f"EGOCENTRIC photoreal GLB, res={res}  (cameras/sec)\n")
    for scene in scenes:
        print(f"--- {scene} ---")
        print(f"{'N':>5} | {'render_batch (multiRT)':>22} | {'render_batch ATLAS':>18} | {'atlas speedup':>13} | {'layered(washed)':>15}")
        for n in Ns:
            a = run("batch", scene, res, n, atlas=False)
            b = run("batch", scene, res, n, atlas=True)
            c = run("layered", scene, res, n, atlas=False)
            sp = (b/a) if (a and b) else 0
            fmt = lambda v: f"{v:.0f}" if v else "n/a"
            print(f"{n:>5} | {fmt(a):>22} | {fmt(b):>18} | {sp:>12.2f}x | {fmt(c):>15}")
        print()


if __name__ == "__main__":
    main()
