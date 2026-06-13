"""THOROUGH warehouse benchmark — the definitive performance picture.

All numbers are env-steps/sec (= N*steps/wall) in the WAREHOUSE-class workload,
delivering observations to torch.cuda. Three honest reference points:

  ours_pbr   : MJWarp GPU physics + Filament PBR (Vulkan zero-copy, optimized
               K=2 wave-sync) IN THE FULL WAREHOUSE (GLB + IBL + 16 spotlights).
               This is mujofil-warp as intended. PHOTOREAL.
  mjwarp_flat: MJWarp GPU physics + its own raycaster. Bare objects only (cannot
               load GLB warehouse). FLAT Lambertian — no PBR/IBL/reflections.
  gl_pbr     : original mujofil OpenGL batched + CPU readback, full warehouse,
               CPU-MuJoCo physics. PHOTOREAL (the prior champion renderer).

Each cell runs in its own subprocess. Reports across N to expose scaling.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
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


def _warehouse(rr, glb_loader):
    rr.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
                os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        glb_loader(os.path.join(A, g))
    rr.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        rr.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)


def worker_ours(res, n, steps, warmup):
    import numpy as np, mujoco, mujoco_warp as mjw, warp as wp, torch
    from mujofil_warp import WarpRenderer, RendererConfig
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
    _warehouse(r, r.load_glb)

    def step():
        mjw.step(M, d); wp.synchronize()
        gx = d.geom_xpos.numpy(); gm = d.geom_xmat.numpy().reshape(n, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]
        obs = r.render_batch(mjm, host, cam_id=0)[..., :3].float()
        torch.cuda.synchronize(); return obs
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(steps): step()
    return n * steps / (time.perf_counter() - t0)


def worker_mjwarp(res, n, steps, warmup):
    import mujoco, mujoco_warp as mjw, warp as wp, torch
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    M = mjw.put_model(mjm); d = mjw.make_data(mjm, nworld=n)
    rc = mjw.create_render_context(mjm, nworld=n, cam_res=(res, res), render_rgb=True,
                                   render_depth=False, use_textures=True, use_shadows=True)
    rgb = wp.zeros((n, res, res), dtype=wp.vec3)

    def step():
        mjw.step(M, d); wp.synchronize()
        mjw.refit_bvh(M, d, rc); mjw.render(M, d, rc); mjw.get_rgb(rc, 0, rgb)
        x = wp.to_torch(rgb); torch.cuda.synchronize(); return x
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(steps): step()
    return n * steps / (time.perf_counter() - t0)


def worker_gl(res, n, steps, warmup):
    import mujoco, torch
    from mujofil import native as vf
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(mjm) for _ in range(n)]
    for dd in datas: mujoco.mj_forward(mjm, dd)
    dptrs = [dd._address for dd in datas]
    c = vf.RendererConfig(); c.width = c.height = res; c.use_vulkan = False
    c.enable_ssao = True; c.enable_shadows = True; c.enable_msaa = True; c.exposure = 1.6; c.vsync = False
    r = vf.Renderer(c); r.initialize(); b = vf.SceneBridge(r); b.load_model(mjm._address)
    b.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
               os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        b.load_glb(os.path.join(A, g))
    b.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        b.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88)
    b.sync_camera(mjm._address, datas[0]._address, 0)

    def step():
        for dd in datas: mujoco.mj_step(mjm, dd)
        rgb = b.render_batch_rgb(mjm._address, dptrs, 0, res, res)
        x = torch.from_numpy(rgb).to("cuda").float(); torch.cuda.synchronize(); return x
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(steps): step()
    return n * steps / (time.perf_counter() - t0)


WORKERS = {"ours": worker_ours, "mjwarp": worker_mjwarp, "gl": worker_gl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(WORKERS), default=None)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()
    if args.worker:
        print("SPS " + json.dumps({"sps": WORKERS[args.worker](args.res, args.n, args.steps, args.warmup)}))
        return

    env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1", MUJOCO_GL="egl")

    def run(worker, res, n, steps=15, warmup=4):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", worker,
               "--res", str(res), "--n", str(n), "--steps", str(steps), "--warmup", str(warmup)]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return float("nan")
        line = [l for l in r.stdout.splitlines() if l.startswith("SPS ")]
        return json.loads(line[0][4:])["sps"] if line else float("nan")

    NS = [int(x) for x in (sys.argv[1:] or [])] if False else [64, 256, 512, 1024, 2048]
    print("\nTHOROUGH WAREHOUSE BENCHMARK — env-steps/sec (-> torch.cuda)")
    print("ours_pbr = MJWarp physics + Filament PBR warehouse (PHOTOREAL, zero-copy)")
    print("mjwarp_flat = MJWarp physics + raycaster (FLAT, objects only)")
    print("gl_pbr = mujofil OpenGL batch warehouse, CPU physics (PHOTOREAL)\n")
    for res in [128, 256]:
        print(f"=== {res}x{res} ===")
        print(f"{'N':>6} {'ours_pbr':>10} {'mjwarp_flat':>13} {'gl_pbr':>10}")
        print("-" * 44)
        for n in NS:
            o = run("ours", res, n)
            mw = run("mjwarp", res, n)
            gl = run("gl", res, n) if n <= 512 else float("nan")  # gl: CPU physics gets slow at huge N
            def fmt(v): return f"{v:>10.0f}" if v == v else f"{'--':>10}"
            print(f"{n:>6} {fmt(o)} {fmt(mw)[:13]:>13} {fmt(gl)}")
        print()
    print("Note: mjwarp_flat renders BARE OBJECTS (no warehouse); ours/gl render the FULL warehouse.")


if __name__ == "__main__":
    main()
