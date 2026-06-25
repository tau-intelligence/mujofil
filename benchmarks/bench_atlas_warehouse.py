"""Warehouse render-only A/B: atlas (megatexture) vs multi-RT single-sync (GL).

This is the scene where per-render-target overhead was measured highest (GLB +
IBL + 16 spotlights + SSAO/shadows). Render-only (no MJWarp physics): render N
static host datas repeatedly so the renderer is isolated. Each cell runs in its
own process (env picks the path), best-of-3, os._exit past the teardown panic.

  MUJOFIL_WARP_BACKEND=gl python benchmarks/bench_atlas_warehouse.py
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
A = os.path.join(WH, "assets")

SCENE = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    <body pos="-0.6 0 0.5"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body pos="0.1 0 0.6"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
    <body pos="0.8 0.3 0.5"><freejoint/><geom type="sphere" size="0.22" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>
  </worldbody>
</mujoco>
"""


def _warehouse(rr):
    rr.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
                os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        rr.load_glb(os.path.join(A, g))
    rr.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        rr.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)


def worker(res, n, iters):
    import numpy as np, mujoco, torch  # noqa: F401
    from mujofil import WarpRenderer, RendererConfig
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    host = [mujoco.MjData(mjm) for _ in range(n)]
    rng = np.random.default_rng(0)
    for i, h in enumerate(host):
        h.qpos[:] = 0
        for b in range(mjm.njnt):
            h.qpos[7 * b + 0] += 0.3 * ((i % 5) - 2)
        mujoco.mj_forward(mjm, h)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.6
    r = WarpRenderer(cfg); r.load_model(mjm)
    _warehouse(r)
    for _ in range(3):
        r.render_batch(mjm, host, cam_id=0)
    torch.cuda.synchronize()
    r.reset_profile()
    t0 = time.perf_counter()
    for _ in range(iters):
        r.render_batch(mjm, host, cam_id=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p = r.profile(); f = max(p["frames"], 1)
    print(f"RESULT n={n} res={res} cam_s={n*iters/dt:.0f} "
          f"render_ms={p['render_ms']/f:.3f} flush_ms={p['flush_ms']/iters:.2f} "
          f"copy_ms={p['copy_ms']/iters:.3f}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(int(sys.argv[i + 1]), int(sys.argv[i + 2]), int(sys.argv[i + 3]))
        return
    res = 256
    Ns = [16, 64, 256]
    iters = 30
    reps = 3
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"=== WAREHOUSE render-only: atlas vs multi-RT 1-sync (GL), res={res}, best-of-{reps} ===")
    modes = [
        ("MULTI 1-sync", {"MUJOFIL_WARP_ATLAS": "0", "MUJOFIL_WARP_FLUSH_EVERY": "100000"}),
        ("ATLAS       ", {"MUJOFIL_WARP_ATLAS": "1"}),
    ]

    def one(n, menv):
        env = dict(os.environ); env["MUJOFIL_WARP_BACKEND"] = "gl"; env.update(menv)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                              str(n), str(res), str(iters)],
                             capture_output=True, text=True, env=env, cwd=repo_root)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
        if not line:
            err = out.stderr.strip().splitlines()
            return None, (err[-1] if err else "(no output)")
        return dict(kv.split("=") for kv in line[7:].split() if "=" in kv), None

    for n in Ns:
        best = {}
        for label, menv in modes:
            top = None
            for _ in range(reps):
                f, err = one(n, menv)
                if f and (top is None or float(f["cam_s"]) > float(top["cam_s"])):
                    top = f
            best[label] = top
            if top:
                print(f"  {label} n={n:<4} cam_s={float(top['cam_s']):.0f} "
                      f"render_ms={float(top['render_ms']):.3f} "
                      f"flush_ms={float(top['flush_ms']):.2f} copy_ms={float(top['copy_ms']):.3f}")
            else:
                print(f"  {label} n={n} FAILED: {err}")
        if best.get("MULTI 1-sync") and best.get("ATLAS       "):
            sp = float(best["ATLAS       "]["cam_s"]) / float(best["MULTI 1-sync"]["cam_s"])
            print(f"    -> ATLAS speedup vs multi-RT single-sync: {sp:.2f}x")


if __name__ == "__main__":
    main()
