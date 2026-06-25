"""A/B: atlas (megatexture, single-RT) vs multi-RT batched GL rendering.

Render-only microbenchmark — same N MjData rendered repeatedly so the renderer
is isolated from physics. Reports cam/s and the render/flush/copy ms breakdown
for each mode across a sweep of batch sizes. Run each mode in its OWN process
(env decides the path) so Filament state never mixes.

  MUJOFIL_WARP_BACKEND=gl python benchmarks/bench_atlas.py            # both modes
  MUJOFIL_WARP_BACKEND=gl MUJOFIL_WARP_ATLAS=1 python ... --worker N  # one cell
"""
import os
import sys
import time
import subprocess

import mujofil
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR",
                      os.path.join(os.path.dirname(mujofil.__file__), "materials"))
# Prefer warp's own materials (match the vendored material_manager).
_warp_mats = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "mujofil", "materials")
if os.path.isdir(_warp_mats):
    os.environ["VF_MUJOCO_MATERIALS_DIR"] = os.path.abspath(_warp_mats)

import math

SCENE_HEAD = """
<mujoco>
  <asset>
    <material name="red"  rgba="0.85 0.25 0.2 1" metallic="0.0" roughness="0.4"/>
    <material name="gold" rgba="0.9 0.7 0.2 1"  metallic="1.0" roughness="0.25"/>
    <material name="blue" rgba="0.2 0.4 0.85 1" metallic="0.3" roughness="0.5"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
    <camera name="cam0" pos="0 -4.5 2.2" xyaxes="1 0 0 0 0.5 0.86"/>
"""


def _scene(nbody=24):
    # Render-bound scene: many PBR bodies so per-frame shading (SSAO/shadows/
    # mesh) is non-trivial -> the per-render-target setup the atlas removes
    # actually matters (a 2-object scene is flush-bound and hides it).
    parts = [SCENE_HEAD]
    mats = ["red", "gold", "blue"]
    for i in range(nbody):
        a = 2 * math.pi * i / nbody
        x, y = 2.2 * math.cos(a), 2.2 * math.sin(a)
        shp, sz = (("box", "0.22 0.22 0.22") if i % 2 else ("sphere", "0.22"))
        parts.append(
            f'<body pos="{x:.3f} {y:.3f} 0.4"><freejoint/>'
            f'<geom type="{shp}" size="{sz}" material="{mats[i % 3]}"/></body>')
    parts.append("</worldbody></mujoco>")
    return "\n".join(parts)


SCENE = _scene()


def run_worker(n, res, iters):
    import numpy as np
    import torch  # noqa: F401  (DLPack consumer)
    import mujoco
    from mujofil import WarpRenderer, RendererConfig

    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for i, d in enumerate(datas):
        mujoco.mj_resetData(m, d)
        for b in range(m.njnt):
            d.qpos[7 * b + 0] += 0.4 * ((i % 5) - 2)  # spread worlds apart
        mujoco.mj_forward(m, d)

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    r = WarpRenderer(cfg)
    r.load_model(m)

    # warmup
    for _ in range(3):
        r.render_batch(m, datas, cam_id=0)
    import torch
    torch.cuda.synchronize()
    r.reset_profile()
    t0 = time.perf_counter()
    for _ in range(iters):
        r.render_batch(m, datas, cam_id=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    prof = r.profile()
    cams = n * iters
    frames = max(prof["frames"], 1)
    print(f"RESULT n={n} res={res} cam_s={cams/dt:.0f} "
          f"render_ms={prof['render_ms']/frames:.3f} "
          f"flush_ms={prof['flush_ms']/iters:.3f} "
          f"copy_ms={prof['copy_ms']/iters:.3f}", flush=True)
    # Skip Python/Filament teardown (a known SceneBridge material-teardown panic
    # aborts the process; harmless here but would lose the flushed RESULT). The
    # OS reclaims the GPU context on exit.
    sys.stdout.flush()
    os._exit(0)


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        n, res, iters = int(sys.argv[i + 1]), int(sys.argv[i + 2]), int(sys.argv[i + 3])
        run_worker(n, res, iters)
        return

    res = 256
    Ns = [16, 64, 256]
    iters = 40
    reps = 3  # best-of: laptop GPU clocks vary; the best run = least-throttled
    print(f"=== atlas vs multi-RT (GL), res={res}, iters={iters}, best-of-{reps}, scene=24 PBR bodies ===")
    modes = [
        ("MULTI 1-sync", {"MUJOFIL_WARP_ATLAS": "0", "MUJOFIL_WARP_FLUSH_EVERY": "100000"}),
        ("ATLAS       ", {"MUJOFIL_WARP_ATLAS": "1"}),
    ]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def one(n, mode_env):
        env = dict(os.environ)
        env["MUJOFIL_WARP_BACKEND"] = "gl"
        env.update(mode_env)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", str(n), str(res), str(iters)],
            capture_output=True, text=True, env=env, cwd=repo_root)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
        if not line:
            err = out.stderr.strip().splitlines()
            return None, (err[-1] if err else "(no output)")
        fields = dict(kv.split("=") for kv in line[7:].split() if "=" in kv)
        return fields, None

    # Interleave modes per N (so throttling hits both equally), best-of-reps.
    for n in Ns:
        best = {}
        for label, mode_env in modes:
            top = None
            for _ in range(reps):
                f, err = one(n, mode_env)
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
