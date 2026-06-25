"""Three-way render-throughput benchmark across batch sizes, resolutions, scenes.

Compares, on the SAME scene / camera / resolution, producing GPU tensors:
  * mujoco    : stock MuJoCo renderer (EGL/OpenGL), N envs rendered SEQUENTIALLY
                (it has no batching) -> uploaded to torch.cuda. The baseline.
  * mjwarp    : MJWarp's batched single-hit raycaster (flat Lambertian, no PBR),
                GPU-resident. The throughput reference (fidelity traded away).
  * layered   : ours -- Filament PBR (IBL/shadows/reflections) via the gl_Layer
                parallel-batch path, zero-copy to torch.cuda.

Metric = cameras/sec (total images per second = N worlds per batched call).
Each measurement runs in its OWN subprocess (clean Filament/CUDA/warp state),
best-of-2. This is an honest head-to-head: MJWarp buys throughput by dropping
shading; we keep photoreal PBR. The numbers show exactly where each lands and
whether our layered path closes the gap to MJWarp's batching.

Run (uncapped address space for CUDA):
  systemd-run --user -p LimitAS=infinity --quiet --wait --pipe -- \
    bash -c 'cd ~/mujofil-warp && source .venv/bin/activate && \
             MUJOFIL_WARP_BACKEND=gl python benchmarks/bench_three_way.py'
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Primitives scene (fair to MJWarp, which prefers primitives; identical geometry
# for all three backends). Materials drive our PBR; MJWarp sees flat colours.
SCENE_PRIM = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="copper" rgba="0.95 0.55 0.35 1" metallic="1.0" roughness="0.35"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
    <material name="floor"  rgba="0.45 0.46 0.50 1" metallic="0.0" roughness="0.6"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"/>
    <body pos="-0.6 0 0.30"><freejoint/><geom type="sphere" size="0.26" material="chrome"/></body>
    <body pos="0.1 0 0.30"><freejoint/><geom type="sphere" size="0.26" material="gold"/></body>
    <body pos="0.8 0 0.30"><freejoint/><geom type="box" size="0.24 0.24 0.24" material="copper"/></body>
    <body pos="-0.2 0.7 0.30"><freejoint/><geom type="sphere" size="0.26" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.1" xyaxes="1 0 0 0 0.34 0.94"/>
  </worldbody>
</mujoco>
"""

FRANKA_XML = "/home/mumuksh/Visual-Fidelity-Mujoco/assets/models/franka_fr3_table.xml"


def _load_model(scene):
    import mujoco
    if scene == "primitives":
        return mujoco.MjModel.from_xml_string(SCENE_PRIM)
    if scene == "franka":
        return mujoco.MjModel.from_xml_path(FRANKA_XML)
    raise ValueError(scene)


def _make_datas(m, nworld):
    import mujoco, numpy as np
    datas = [mujoco.MjData(m) for _ in range(nworld)]
    for i, d in enumerate(datas):
        k = min(7, m.nq)
        if k:
            d.qpos[:k] += 0.02 * np.sin(np.arange(k) + i)
        mujoco.mj_forward(m, d)
    return datas


# ---------------------------------------------------------------------------
def worker_mujoco(scene, res, iters, warmup, nworld):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np, mujoco, torch
    m = _load_model(scene)
    datas = _make_datas(m, nworld)
    r = mujoco.Renderer(m, height=res, width=res)
    cam = 0 if m.ncam > 0 else -1

    def step():
        out = np.empty((nworld, res, res, 3), dtype=np.uint8)
        for i, d in enumerate(datas):
            r.update_scene(d, camera=cam if cam >= 0 else -1)
            out[i] = r.render()
        t = torch.from_numpy(out).to("cuda")
        torch.cuda.synchronize()
        return t

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return nworld * iters / (time.perf_counter() - t0)


def worker_mjwarp(scene, res, iters, warmup, nworld):
    import mujoco, mujoco_warp as mjw, warp as wp, torch
    mjm = _load_model(scene)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    rc = mjw.create_render_context(
        mjm, nworld=nworld, cam_res=(res, res), render_rgb=True,
        render_depth=False, use_textures=True, use_shadows=True)
    rgb = wp.zeros((nworld, res, res), dtype=wp.vec3)

    def step():
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        mjw.get_rgb(rc, 0, rgb)
        t = wp.to_torch(rgb)
        torch.cuda.synchronize()
        return t

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return nworld * iters / (time.perf_counter() - t0)


def worker_layered(scene, res, iters, warmup, nworld):
    import mujoco, torch
    from mujofil import WarpRenderer, RendererConfig
    m = _load_model(scene)
    datas = _make_datas(m, nworld)
    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.batch_size = nworld; cfg.layered = True
    r = WarpRenderer(cfg); r.load_model(m)
    cam = 0 if m.ncam > 0 else -1
    if cam < 0:
        r.set_free_camera(0, -3.0, 1.2, 0, 0, 0.4)

    def step():
        t = r.render_batch_layered(m, datas, cam_id=cam)[..., :3]
        torch.cuda.synchronize()
        return t

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    dt = time.perf_counter() - t0
    r.close()
    return nworld * iters / dt


WORKERS = {"mujoco": worker_mujoco, "mjwarp": worker_mjwarp, "layered": worker_layered}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(WORKERS))
    ap.add_argument("--scene", default="primitives")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--nworld", type=int, default=1)
    args = ap.parse_args()

    if args.worker:
        fps = WORKERS[args.worker](args.scene, args.res, args.iters,
                                   args.warmup, args.nworld)
        print("FPS " + json.dumps({"fps": fps}))
        return

    env = dict(os.environ, MUJOFIL_WARP_BACKEND="gl", MUJOFIL_NO_DRIVER_WARNING="1")

    def run(worker, scene, res, nworld, iters, warmup):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", worker,
               "--scene", scene, "--res", str(res), "--iters", str(iters),
               "--warmup", str(warmup), "--nworld", str(nworld)]
        best = float("nan")
        for _ in range(2):
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("FPS ")]
            if line:
                v = json.loads(line[0][4:])["fps"]
                best = v if best != best else max(best, v)
        return best

    SCENES = ["primitives", "franka"]
    RES = [128, 256]
    NWORLD = [1, 16, 64, 256]

    for scene in SCENES:
        print(f"\n########## SCENE = {scene} ##########")
        for res in RES:
            print(f"\n--- res {res}x{res}  (cameras/sec) ---")
            print(f"{'backend':>10} | " + " ".join(f"N={n:<6}" for n in NWORLD))
            print("-" * 58)
            for worker in ["mujoco", "mjwarp", "layered"]:
                it, wu = (40, 8) if res <= 128 else (24, 6)
                cells = []
                for n in NWORLD:
                    cells.append(run(worker, scene, res, n, it, wu))
                print(f"{worker:>10} | " +
                      " ".join(f"{c:>8.0f}" for c in cells))
            # speedups vs the others at the largest batch
            print(f"  (cameras/sec; mjwarp=flat raycast no-PBR, layered=ours full PBR)")

    print("\nLegend: mujoco=stock EGL renderer, sequential N (no batching). "
          "mjwarp=batched flat raycaster (no PBR). layered=ours, batched PBR.")


if __name__ == "__main__":
    main()
