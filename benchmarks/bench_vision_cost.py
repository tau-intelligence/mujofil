"""Vision-cost benchmark -- answers the three questions a reviewer raised:

  Q1. "Throughput goes UP as N grows -- that seems backwards."
      -> We report BOTH cameras/sec AND milliseconds-per-batched-frame. The
         per-frame latency rises sublinearly (fixed per-batch overhead -- one
         sync, kernel launches -- amortizes over more worlds), so throughput
         climbs then plateaus once it's render-bound. Nothing is free; the
         latency column makes that explicit.

  Q2. "Have you compared RGB to just depth? People train on depth, not RGB."
      -> We time MJWarp's native DEPTH raycaster (the cheap incumbent) and its
         flat RGB raycaster against our photoreal PBR RGB, same scene/N/res.

  Q3. "How much does performance drop with vision enabled vs no vision?"
      -> `physics_only` runs the identical GPU physics loop with NO rendering.
         Every render mode's throughput is reported as a fraction of it = the
         exact rendering tax.

Metric: env-steps/sec = N * iters / wall  (one env-step = one physics step, plus
one batched render of all N worlds for the vision modes). Also ms/step.

Each cell runs in its OWN subprocess (clean CUDA/Filament/warp state), best-of-2.

Run (uncapped address space for CUDA):
  systemd-run --user -p LimitAS=infinity --quiet --wait --pipe -- \
    bash -c 'cd ~/mujofil-warp && source .venv/bin/activate && \
             MUJOFIL_WARP_BACKEND=gl python benchmarks/bench_vision_cost.py'
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# A small RL-like scene: a floor + a few falling bodies with PBR materials and a
# fixed camera. Identical geometry for every backend (fair comparison); the
# materials only matter to our PBR path, MJWarp sees flat colours / depth.
SCENE = """
<mujoco>
  <option timestep="0.004"/>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
    <material name="floor"  rgba="0.45 0.46 0.50 1" metallic="0.0" roughness="0.6"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"/>
    <body pos="-0.6 0 0.8"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body pos="0.1 0 1.0"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
    <body pos="0.8 0.3 0.7"><freejoint/><geom type="sphere" size="0.22" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>
  </worldbody>
</mujoco>
"""


def _model():
    import mujoco
    return mujoco.MjModel.from_xml_string(SCENE)


# --- workers (each returns (env_steps_per_sec, ms_per_step)) --------------------

def _result(nworld, iters, dt):
    return nworld * iters / dt, dt / iters * 1000.0


def worker_physics_only(res, iters, warmup, nworld):
    """GPU physics, NO rendering -- the 'no vision' baseline."""
    import mujoco_warp as mjw, warp as wp
    mjm = _model()
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)

    def step():
        mjw.step(m, d)
        wp.synchronize()

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return _result(nworld, iters, time.perf_counter() - t0)


def worker_mjwarp_depth(res, iters, warmup, nworld):
    """GPU physics + MJWarp's native single-hit DEPTH raycaster."""
    import mujoco_warp as mjw, warp as wp, torch
    mjm = _model()
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    rc = mjw.create_render_context(
        mjm, nworld=nworld, cam_res=(res, res), render_rgb=False,
        render_depth=True, use_textures=False, use_shadows=False)
    depth = wp.zeros((nworld, res, res), dtype=float)

    def step():
        mjw.step(m, d)
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        mjw.get_depth(rc, 0, 1.0, depth)
        _ = wp.to_torch(depth)
        torch.cuda.synchronize()

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return _result(nworld, iters, time.perf_counter() - t0)


def worker_mjwarp_rgb(res, iters, warmup, nworld):
    """GPU physics + MJWarp's flat-Lambertian RGB raycaster."""
    import mujoco_warp as mjw, warp as wp, torch
    mjm = _model()
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    rc = mjw.create_render_context(
        mjm, nworld=nworld, cam_res=(res, res), render_rgb=True,
        render_depth=False, use_textures=True, use_shadows=True)
    rgb = wp.zeros((nworld, res, res), dtype=wp.vec3)

    def step():
        mjw.step(m, d)
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        mjw.get_rgb(rc, 0, rgb)
        _ = wp.to_torch(rgb)
        torch.cuda.synchronize()

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return _result(nworld, iters, time.perf_counter() - t0)


def worker_ours_rgb(res, iters, warmup, nworld):
    """GPU physics + our photoreal Filament PBR RGB, zero-copy to torch.cuda.

    This is exactly the shipped ParallelScene path: GPU physics, pull per-world
    geom transforms to host MjData, batched PBR render. Includes the transform
    host-copy so the cost is honest (the pixels are zero-copy, the transforms
    are not -- yet)."""
    import mujoco, mujoco_warp as mjw, warp as wp, torch
    from mujofil import WarpRenderer
    mjm = _model()
    ngeom = mjm.ngeom
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    host = [mujoco.MjData(mjm) for _ in range(nworld)]
    r = WarpRenderer(width=res, height=res, batch_size=nworld, preset="train")
    r.load_model(mjm)

    def step():
        mjw.step(m, d)
        wp.synchronize()
        gx = d.geom_xpos.numpy()
        gm = d.geom_xmat.numpy().reshape(nworld, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]
            h.geom_xmat[:] = gm[i]
        _ = r.render_batch(mjm, host, cam_id=0)
        torch.cuda.synchronize()

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    dt = time.perf_counter() - t0
    r.close()
    return _result(nworld, iters, dt)


def worker_ours_layered(res, iters, warmup, nworld):
    """GPU physics + our LAYERED path: all N worlds in ONE instanced gl_Layer
    draw (effects off in the objects pass, shared/view-folded camera). The
    high-throughput path, zero-copy to torch.cuda."""
    import mujoco, mujoco_warp as mjw, warp as wp, torch
    from mujofil import WarpRenderer, RendererConfig
    mjm = _model()
    ngeom = mjm.ngeom
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    host = [mujoco.MjData(mjm) for _ in range(nworld)]
    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.batch_size = nworld; cfg.layered = True
    r = WarpRenderer(cfg); r.load_model(mjm)

    def step():
        mjw.step(m, d)
        wp.synchronize()
        gx = d.geom_xpos.numpy()
        gm = d.geom_xmat.numpy().reshape(nworld, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]
            h.geom_xmat[:] = gm[i]
        _ = r.render_batch_layered(mjm, host, cam_id=0)
        torch.cuda.synchronize()

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    dt = time.perf_counter() - t0
    r.close()
    return _result(nworld, iters, dt)


WORKERS = {
    "physics_only": worker_physics_only,
    "mjwarp_depth": worker_mjwarp_depth,
    "mjwarp_rgb": worker_mjwarp_rgb,
    "ours_rgb": worker_ours_rgb,
    "ours_layered": worker_ours_layered,
}
ORDER = ["physics_only", "mjwarp_depth", "mjwarp_rgb", "ours_rgb", "ours_layered"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(WORKERS))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--res-list", type=str, default="128,256")
    ap.add_argument("--n-list", type=str, default="16,64,256,512,1024")
    args = ap.parse_args()

    if args.worker:
        sps, ms = WORKERS[args.worker](args.res, args.iters, args.warmup, args.nworld)
        print("RES " + json.dumps({"sps": sps, "ms": ms}))
        return

    env = dict(os.environ, MUJOFIL_WARP_BACKEND="gl", MUJOFIL_NO_DRIVER_WARNING="1")
    RES = [int(x) for x in args.res_list.split(",")]
    NWORLD = [int(x) for x in args.n_list.split(",")]

    def run(worker, res, nworld):
        it, wu = (40, 8) if res <= 128 else (24, 6)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", worker,
               "--res", str(res), "--iters", str(it), "--warmup", str(wu),
               "--nworld", str(nworld)]
        best = None
        for _ in range(2):
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
            if line:
                v = json.loads(line[0][4:])
                if best is None or v["sps"] > best["sps"]:
                    best = v
            elif best is None:
                sys.stderr.write(p.stderr[-600:] + "\n")
        return best

    results = {}
    for res in RES:
        print(f"\n################  res {res}x{res}  ################")
        # throughput table
        print("\nenv-steps/sec  (one step = physics + 1 batched render of N worlds)")
        print(f"{'mode':>14} | " + " ".join(f"N={n:<7}" for n in NWORLD))
        print("-" * (17 + 10 * len(NWORLD)))
        for worker in ORDER:
            cells = []
            for n in NWORLD:
                r = run(worker, res, n)
                results[(res, worker, n)] = r
                cells.append(f"{r['sps']:<9.0f}" if r else "--       ")
            print(f"{worker:>14} | " + " ".join(cells))

        # latency table (demystifies "throughput rises with N")
        print("\nmilliseconds per step  (per batched frame -- rises SUBLINEARLY in N)")
        print(f"{'mode':>14} | " + " ".join(f"N={n:<7}" for n in NWORLD))
        print("-" * (17 + 10 * len(NWORLD)))
        for worker in ORDER:
            cells = []
            for n in NWORLD:
                r = results.get((res, worker, n))
                cells.append(f"{r['ms']:<9.2f}" if r else "--       ")
            print(f"{worker:>14} | " + " ".join(cells))

        # the two derived answers
        print("\nrendering tax  (throughput as a fraction of physics_only = 'no vision')")
        print(f"{'mode':>14} | " + " ".join(f"N={n:<7}" for n in NWORLD))
        print("-" * (17 + 10 * len(NWORLD)))
        base = {n: results.get((res, "physics_only", n)) for n in NWORLD}
        for worker in ["mjwarp_depth", "mjwarp_rgb", "ours_rgb", "ours_layered"]:
            cells = []
            for n in NWORLD:
                r = results.get((res, worker, n)); b = base[n]
                cells.append(f"{r['sps']/b['sps']:<9.2f}" if r and b else "--       ")
            print(f"{worker:>14} | " + " ".join(cells))

    print("\nDONE")


if __name__ == "__main__":
    main()
