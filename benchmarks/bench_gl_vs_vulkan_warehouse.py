"""Definitive GL single-sync vs Vulkan zero-copy benchmark — WAREHOUSE + MJWarp.

Both paths: MJWarp GPU physics (mjw.step) -> pull geom transforms -> Filament PBR
render of the FULL warehouse (GLB + IBL + 16 spotlights) -> torch.cuda (zero-copy).
The ONLY difference is the renderer backend:

  gl     : OpenGL/GLX, N renders bracketed by ONE beginFrame/endFrame + ONE
           flushAndWait = TRUE single-sync (the prize). MUJOFIL_BACKEND=gl.
  vulkan : shared Vulkan device + exportable swapchain, K=2 wave-sync (forced by
           Filament's 2-frame in-flight cap). MUJOFIL_BACKEND=vulkan.

Reports env-steps/s (= N*steps/wall) AND the render/flush/copy profile breakdown,
across N up to 2048, subprocess-isolated per cell.
"""
import argparse
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


def worker(backend, res, n, steps, warmup):
    import numpy as np, mujoco, mujoco_warp as mjw, warp as wp, torch
    from mujofil import WarpRenderer, RendererConfig
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    M = mjw.put_model(mjm); d = mjw.make_data(mjm, nworld=n)
    qp = d.qpos.numpy(); qp[:] += np.random.default_rng(0).uniform(-0.05, 0.05, qp.shape)
    wp.copy(d.qpos, wp.array(qp, dtype=float))
    host = [mujoco.MjData(mjm) for _ in range(n)]
    for h in host: mujoco.mj_forward(mjm, h)
    ngeom = mjm.ngeom
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.6
    r = WarpRenderer(cfg); r.load_model(mjm)
    _warehouse(r)

    def step():
        mjw.step(M, d); wp.synchronize()
        gx = d.geom_xpos.numpy(); gm = d.geom_xmat.numpy().reshape(n, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]
        obs = r.render_batch(mjm, host, cam_id=0)[..., :3].float()
        torch.cuda.synchronize(); return obs

    for _ in range(warmup): step()
    r.reset_profile()
    t0 = time.perf_counter()
    for _ in range(steps): step()
    dt = time.perf_counter() - t0
    prof = r.profile()
    f = max(prof["frames"], 1)
    return {
        "env_sps": n * steps / dt,
        "cam_sps": n * steps / dt,
        "render_ms": prof["render_ms"] / steps,
        "flush_ms": prof["flush_ms"] / steps,
        "copy_ms": prof["copy_ms"] / steps,
        "wall_ms": dt / steps * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gl", "vulkan"], default=None)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()
    if args.backend:
        print("RES " + json.dumps(worker(args.backend, args.res, args.n, args.steps, args.warmup)))
        return

    def run(backend, res, n, steps=15, warmup=4):
        env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1", MUJOCO_GL="egl",
                   MUJOFIL_BACKEND=backend,
                   MUJOFIL_WARP_FLUSH_EVERY=("100000" if backend == "gl" else "2"))
        cmd = [sys.executable, os.path.abspath(__file__), "--backend", backend,
               "--res", str(res), "--n", str(n), "--steps", str(steps), "--warmup", str(warmup)]
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return None
        line = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
        if not line:
            sys.stderr.write(p.stdout[-2000:] + "\n" + p.stderr[-2000:] + "\n")
            return None
        return json.loads(line[0][4:])

    NS = [64, 256, 512, 1024, 2048]
    print("\nGL SINGLE-SYNC vs VULKAN — WAREHOUSE + MJWarp GPU physics -> torch.cuda")
    print("env-steps/s (= cam/s here, 1 cam/world) + render/flush/copy ms per step\n")
    for res in [128, 256]:
        print(f"=== {res}x{res} ===")
        print(f"{'N':>5} | {'gl cam/s':>9} {'gl flush':>9} {'gl copy':>8} | "
              f"{'vk cam/s':>9} {'vk flush':>9} | {'speedup':>7}")
        print("-" * 78)
        for n in NS:
            g = run("gl", res, n)
            v = run("vulkan", res, n)
            gs = g["cam_sps"] if g else float("nan")
            vs = v["cam_sps"] if v else float("nan")
            gf = g["flush_ms"] if g else float("nan")
            gc = g["copy_ms"] if g else float("nan")
            vf = v["flush_ms"] if v else float("nan")
            sp = (gs / vs) if (g and v and vs) else float("nan")
            def f(x, w=9, p=0): return f"{x:>{w}.{p}f}" if x == x else f"{'--':>{w}}"
            print(f"{n:>5} | {f(gs)} {f(gf,9,1)} {f(gc,8,2)} | {f(vs)} {f(vf,9,1)} | {f(sp,7,2)}")
        print()


if __name__ == "__main__":
    main()
