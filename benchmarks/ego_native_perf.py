"""The decisive test: egocentric over a NATIVE-geometry environment, layered
single-draw vs render_batch loop. Earlier we saw egocentric render_batch is
OVERHEAD-bound (heavy GLB == no backdrop), so collapsing N views into ONE
instanced draw should win. This measures exactly that real tiled benefit.
"""
import os, sys, time
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)
import numpy as np, torch, mujoco
from mujofil import WarpRenderer, RendererConfig

# A NATIVE enclosed room (walls/ceiling/floor/furniture) -- the environment is
# real geometry, not a GLB backdrop, so the layered path can instance it per world.
SCENE = """
<mujoco><option timestep="0.01"/>
  <asset>
    <material name="floor" rgba="0.55 0.42 0.30 1" roughness="0.5"/>
    <material name="wall"  rgba="0.82 0.80 0.76 1" roughness="0.9"/>
    <material name="ceil"  rgba="0.90 0.90 0.92 1" roughness="0.95"/>
    <material name="red"   rgba="0.80 0.20 0.18 1" roughness="0.45"/>
    <material name="green" rgba="0.20 0.62 0.32 1" roughness="0.45"/>
    <material name="blue"  rgba="0.20 0.40 0.80 1" roughness="0.45"/>
    <material name="gold"  rgba="0.90 0.74 0.36 1" roughness="0.18" metallic="0.9"/>
    <material name="chrome" rgba="0.85 0.86 0.90 1" roughness="0.12" metallic="0.95"/>
  </asset>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.05" size="6 6 0.05" material="floor"/>
    <geom name="ceil"  type="box" pos="0 0 4" size="6 6 0.05" material="ceil"/>
    <geom type="box" pos="6 0 2" size="0.05 6 2" material="wall"/>
    <geom type="box" pos="-6 0 2" size="0.05 6 2" material="wall"/>
    <geom type="box" pos="0 6 2" size="6 0.05 2" material="wall"/>
    <geom type="box" pos="0 -6 2" size="6 0.05 2" material="wall"/>
    <geom type="cylinder" pos="0 0 0.4" size="0.8 0.4" material="gold"/>
    <geom type="sphere" pos="0 0 1.2" size="0.6" material="chrome"/>
    <geom type="box" pos="4.6 0 0.5" size="0.5 0.7 0.5" material="red"/>
    <geom type="box" pos="-4.6 0 0.6" size="0.6 0.6 0.6" material="green"/>
    <geom type="cylinder" pos="0 4.6 0.7" size="0.5 0.7" material="blue"/>
    <geom type="sphere" pos="3.2 3.2 0.5" size="0.5" material="gold"/>
    <light pos="0 0 3.8" dir="0 0 -1"/>
    <light pos="3 3 3.6" dir="-0.4 -0.4 -1"/>
    <body name="robot" pos="3 0 0.35"><freejoint/>
      <geom type="box" size="0.3 0.2 0.15" rgba="0.12 0.12 0.14 1"/>
      <camera name="ego" pos="0.3 0 0.2" xyaxes="0 -1 0 0.18 0 0.98"/>
    </body>
  </worldbody></mujoco>"""


def make(n, res, layered):
    m = mujoco.MjModel.from_xml_string(SCENE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    jadr = m.jnt_qposadr[0]
    datas = []
    for i in range(n):
        d = mujoco.MjData(m); a = 2*np.pi*i/n
        d.qpos[jadr:jadr+3] = [2.6*np.cos(a), 2.6*np.sin(a), 0.35]
        h = a+np.pi; d.qpos[jadr+3:jadr+7] = [np.cos(h/2), 0, 0, np.sin(h/2)]
        mujoco.mj_forward(m, d); datas.append(d)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_shadows = True; cfg.layered = layered
    r = WarpRenderer(cfg); r.load_model(m)
    return m, datas, cam, r


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    Ns = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["16", "64", "256"])]
    print(f"EGOCENTRIC over NATIVE geometry, res={res}  (cameras/sec, higher=better)\n")
    print(f"{'N':>5} | {'render_batch EGO':>16} | {'layered EGO (1 draw)':>20} | {'tiled speedup':>13}")
    for n in Ns:
        m, d, cam, r = make(n, res, layered=False)
        a = timed(lambda: r.render_batch(m, d, cam_id=cam)); r.close()
        m2, d2, cam2, r2 = make(n, res, layered=True)
        b = timed(lambda: r2.render_batch_layered(m2, d2, cam_id=cam2)); r2.close()
        print(f"{n:>5} | {n/a:>16.0f} | {n/b:>20.0f} | {(n/b)/(n/a):>12.1f}x")


if __name__ == "__main__":
    main()
