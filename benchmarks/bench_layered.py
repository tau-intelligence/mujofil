"""Benchmark: LAYERED one-pass vs per-world render_batch (same scene, -> torch.cuda).

Each mode in its own subprocess (clean Filament state), best-of-3, os._exit past
teardown. Reports cam/s + render/flush/copy breakdown.
"""
import os
import sys
import time
import subprocess

import mujofil
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR",
                      os.path.join(os.path.dirname(mujofil.__file__), "materials"))
# Prefer warp's own materials (match the vendored material_manager; the per-world
# path needs the emissive uniform the installed mujofil 0.1.0 lacks).
_warp_mats = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "mujofil_warp", "materials")
if os.path.isdir(_warp_mats):
    os.environ["VF_MUJOCO_MATERIALS_DIR"] = _warp_mats

SCENE = """
<mujoco>
  <asset><material name="red" rgba="0.85 0.25 0.2 1" metallic="0.0" roughness="0.4"/>
         <material name="gold" rgba="0.9 0.7 0.2 1" metallic="1.0" roughness="0.3"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
    <body pos="0 0 0.4"><freejoint/><geom type="box" size="0.25 0.25 0.25" material="red"/></body>
    <body pos="0.7 0.3 0.3"><freejoint/><geom type="sphere" size="0.2" material="gold"/></body>
    <camera name="cam0" pos="0 -2.6 1.0" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>
"""


def worker(mode, res, n, iters):
    import numpy as np, torch, mujoco
    from mujofil_warp import WarpRenderer, RendererConfig
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for i, d in enumerate(datas):
        d.qpos[0] = -1.5 + 3.0 * i / max(n - 1, 1)
        mujoco.mj_forward(m, d)
    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    if mode == "layered":
        cfg.layered = True
        cfg.enable_msaa = False
    r = WarpRenderer(cfg)
    r.load_model(m)
    fn = r.render_batch_layered if mode == "layered" else r.render_batch
    for _ in range(3):
        fn(m, datas, cam_id=0)
    torch.cuda.synchronize()
    r.reset_profile()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(m, datas, cam_id=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p = r.profile()
    print(f"RESULT cam_s={n*iters/dt:.0f} render_ms={p['render_ms']/iters:.3f} "
          f"flush_ms={p['flush_ms']/iters:.3f} copy_ms={p['copy_ms']/iters:.3f}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i+1], int(sys.argv[i+2]), int(sys.argv[i+3]), int(sys.argv[i+4]))
        return
    res = 128
    Ns = [16, 64, 256]
    iters = 30
    reps = 3
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"=== LAYERED one-pass vs per-world render_batch (GL), res={res}, best-of-{reps} ===")

    def one(mode, n):
        env = dict(os.environ); env["MUJOFIL_WARP_BACKEND"] = "gl"
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", mode,
                              str(res), str(n), str(iters)],
                             capture_output=True, text=True, env=env, cwd=root)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
        if not line:
            return None, (out.stderr.strip().splitlines() or ["(no output)"])[-1]
        return dict(kv.split("=") for kv in line[7:].split() if "=" in kv), None

    for n in Ns:
        best = {}
        for mode in ("per-world", "layered"):
            top = None
            for _ in range(reps):
                f, err = one(mode, n)
                if f and (top is None or float(f["cam_s"]) > float(top["cam_s"])):
                    top = f
            best[mode] = top
            if top:
                print(f"  {mode:10s} n={n:<4} cam_s={float(top['cam_s']):.0f} "
                      f"render_ms={float(top['render_ms']):.3f} flush_ms={float(top['flush_ms']):.3f} "
                      f"copy_ms={float(top['copy_ms']):.3f}")
            else:
                print(f"  {mode:10s} n={n} FAILED: {err}")
        if best.get("per-world") and best.get("layered"):
            sp = float(best["layered"]["cam_s"]) / float(best["per-world"]["cam_s"])
            print(f"    -> LAYERED speedup: {sp:.2f}x")


if __name__ == "__main__":
    main()
