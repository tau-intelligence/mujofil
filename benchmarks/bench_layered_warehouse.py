"""Warehouse throughput: LAYERED one-pass vs per-world render_batch.

Both render the full photoreal warehouse (GLB + IBL + spotlights) + per-world
objects -> torch.cuda. Layered does it in ONE instanced pass + backdrop composite;
per-world renders the whole warehouse N times serially. Subprocess per cell,
best-of-3.
"""
import os
import sys
import time
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujofil
_warp_mats = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "mujofil", "materials")
if os.path.isdir(_warp_mats):
    os.environ["VF_MUJOCO_MATERIALS_DIR"] = _warp_mats

WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
A = os.path.join(WH, "assets")

SCENE = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.06"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.20"/>
    <material name="copper" rgba="0.95 0.6 0.45 1"  metallic="1.0" roughness="0.30"/>
    <material name="red"    rgba="0.80 0.18 0.15 1" metallic="0.0" roughness="0.55"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    <body pos="-0.6 0 0.45"><freejoint/><geom type="sphere" size="0.30" material="chrome"/></body>
    <body pos="0.1 -0.3 0.30"><freejoint/><geom type="box" size="0.24 0.24 0.24" material="gold"/></body>
    <body pos="0.6 0.3 0.26"><freejoint/><geom type="sphere" size="0.22" material="copper"/></body>
    <body pos="0.0 0.6 0.20"><freejoint/><geom type="capsule" size="0.12 0.18" material="red"/></body>
    <camera name="cam0" pos="0 -3.0 1.2" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>
"""


def _warehouse(r):
    r.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
               os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        r.load_glb(os.path.join(A, g))
    r.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        r.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)


def worker(mode, res, n, iters):
    import numpy as np, torch, mujoco
    from mujofil import WarpRenderer, RendererConfig
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for i, d in enumerate(datas):
        for b in range(m.njnt):
            d.qpos[7 * b + 0] += 0.25 * ((i % 5) - 2)
        mujoco.mj_forward(m, d)
    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    cfg.exposure = 1.6
    if mode == "layered":
        cfg.layered = True
        cfg.enable_msaa = False
    r = WarpRenderer(cfg)
    r.load_model(m)
    _warehouse(r)
    fn = r.render_batch_layered if mode == "layered" else r.render_batch
    for _ in range(3):
        fn(m, datas, cam_id=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(m, datas, cam_id=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"RESULT cam_s={n*iters/dt:.0f}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i+1], int(sys.argv[i+2]), int(sys.argv[i+3]), int(sys.argv[i+4]))
        return
    res = 256
    Ns = [16, 64, 256]
    iters = 20
    reps = 3
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"=== WAREHOUSE: layered one-pass vs per-world (GL), res={res}, best-of-{reps} ===")

    def one(mode, n):
        env = dict(os.environ); env["MUJOFIL_BACKEND"] = "gl"
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", mode,
                              str(res), str(n), str(iters)],
                             capture_output=True, text=True, env=env, cwd=root)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
        if not line:
            return None, (out.stderr.strip().splitlines() or ["(no output)"])[-1]
        return float(line.split("=")[1]), None

    for n in Ns:
        best = {}
        for mode in ("per-world", "layered"):
            top = None
            for _ in range(reps):
                v, err = one(mode, n)
                if v and (top is None or v > top):
                    top = v
            best[mode] = top
            print(f"  {mode:10s} n={n:<4} {top:.0f} cam/s" if top else f"  {mode} n={n} FAIL: {err}")
        if best.get("per-world") and best.get("layered"):
            print(f"    -> LAYERED speedup: {best['layered']/best['per-world']:.2f}x")


if __name__ == "__main__":
    main()
