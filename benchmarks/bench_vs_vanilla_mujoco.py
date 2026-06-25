"""Head-to-head: mujofil-warp GL (full warehouse PBR) vs VANILLA MuJoCo (flat).

The bar the user set: "call it a win if we achieve numbers close to vanilla MuJoCo".

Both paths use IDENTICAL MJWarp GPU physics (mjw.step) so only the RENDERER differs:

  ours_gl : mujofil-warp OpenGL single-sync zero-copy. Renders the FULL warehouse
            (3 GLB meshes + IBL + 16 spotlights = ~100x more geometry) with PBR,
            SSAO, shadows, MSAA. Pixels stay on GPU -> torch.cuda (zero-copy).
  mujoco  : stock mujoco.Renderer (MUJOCO_GL=egl). Flat Gouraud shading. CANNOT
            load the warehouse GLB, so it renders the BARE objects only. Each obs
            is read back GPU->CPU (numpy) then uploaded CPU->GPU to torch.cuda.

So ours does FAR more work per frame. Matching vanilla MuJoCo's cam/s = a clear win.
env-steps/s (= cam/s), subprocess-isolated per cell, N up to 2048.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.bench_gl_vs_vulkan_warehouse import SCENE, _warehouse  # noqa: E402


def worker_ours(res, n, steps, warmup):
    import numpy as np, mujoco, mujoco_warp as mjw, warp as wp, torch
    from mujofil import WarpRenderer, RendererConfig
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    M = mjw.put_model(mjm); d = mjw.make_data(mjm, nworld=n)
    host = [mujoco.MjData(mjm) for _ in range(n)]
    for h in host: mujoco.mj_forward(mjm, h)
    ng = mjm.ngeom
    bare = os.environ.get("OURS_BARE", "0") == "1"
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.6
    cfg.enable_ssao = not bare  # bare/fair-vs-MuJoCo mode drops SSAO (MuJoCo has none)
    r = WarpRenderer(cfg); r.load_model(mjm)
    if not bare:
        _warehouse(r)  # full photoreal warehouse (the capability MuJoCo lacks)

    def step():
        mjw.step(M, d); wp.synchronize()
        gx = d.geom_xpos.numpy(); gm = d.geom_xmat.numpy().reshape(n, ng, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]
        obs = r.render_batch(mjm, host, cam_id=0)[..., :3].float()
        torch.cuda.synchronize(); return obs
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(steps): step()
    return {"cam_sps": n * steps / (time.perf_counter() - t0)}


def worker_mujoco(res, n, steps, warmup):
    import numpy as np, mujoco, mujoco_warp as mjw, warp as wp, torch
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    M = mjw.put_model(mjm); d = mjw.make_data(mjm, nworld=n)
    host = [mujoco.MjData(mjm) for _ in range(n)]
    for h in host: mujoco.mj_forward(mjm, h)
    ng = mjm.ngeom
    ren = mujoco.Renderer(mjm, height=res, width=res)
    cam = 0

    def step():
        mjw.step(M, d); wp.synchronize()
        gx = d.geom_xpos.numpy(); gm = d.geom_xmat.numpy().reshape(n, ng, 9)
        frames = np.empty((n, res, res, 3), dtype=np.uint8)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]
            ren.update_scene(h, camera=cam)
            frames[i] = ren.render()
        obs = torch.from_numpy(frames).to("cuda", non_blocking=True).float()
        torch.cuda.synchronize(); return obs
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(steps): step()
    return {"cam_sps": n * steps / (time.perf_counter() - t0)}


WORKERS = {"ours": worker_ours, "mujoco": worker_mujoco}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(WORKERS), default=None)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()
    if args.worker:
        print("RES " + json.dumps(WORKERS[args.worker](args.res, args.n, args.steps, args.warmup)))
        return

    def run(worker, res, n, steps=15, warmup=4):
        env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1", MUJOCO_GL="egl",
                   MUJOFIL_BACKEND="gl", MUJOFIL_WARP_FLUSH_EVERY="100000")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", worker,
               "--res", str(res), "--n", str(n), "--steps", str(steps), "--warmup", str(warmup)]
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return None
        line = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
        if not line:
            sys.stderr.write((p.stdout[-1500:] + "\n" + p.stderr[-1500:]) + "\n")
            return None
        return json.loads(line[0][4:])

    NS = [64, 256, 512, 1024, 2048]
    print("\nmujofil-warp GL (FULL WAREHOUSE PBR, zero-copy) vs VANILLA MuJoCo (flat objects)")
    print("Same MJWarp GPU physics. env-steps/s (= cam/s) -> torch.cuda.\n")
    for res in [128, 256]:
        print(f"=== {res}x{res} ===")
        print(f"{'N':>6} {'ours_gl':>10} {'mujoco':>10} {'ratio':>8}   (ours renders the whole warehouse)")
        print("-" * 60)
        for n in NS:
            o = run("ours", res, n)
            m = run("mujoco", res, n)
            os_ = o["cam_sps"] if o else float("nan")
            ms = m["cam_sps"] if m else float("nan")
            ratio = (os_ / ms) if (o and m and ms) else float("nan")
            def f(x, w=10, p=0): return f"{x:>{w}.{p}f}" if x == x else f"{'--':>{w}}"
            print(f"{n:>6} {f(os_)} {f(ms)} {f(ratio,8,2)}")
        print()
    print("ours_gl = MJWarp physics + Filament PBR FULL WAREHOUSE + zero-copy to torch.cuda")
    print("mujoco  = MJWarp physics + stock mujoco.Renderer (flat, bare objects) + CPU bounce")


if __name__ == "__main__":
    main()
