"""Parallel Vision-RL throughput in the WAREHOUSE environment.

Same as bench_rl.py but OURS renders inside the full photoreal warehouse (GLB
floor/walls/racks + IBL + 16 ceiling spotlights) — the real heavy workload. The
flat renderers (stock MuJoCo, MJWarp) CANNOT load GLB scenes, so they render the
bare MuJoCo objects only. This is an honest asymmetry: ours draws ~100x more
geometry (the whole warehouse) every frame while still delivering photoreal,
zero-copy observations. The comparison shows what that photorealism costs.

One RL step = physics + render N obs + CNN fwd/bwd. Report env-steps/sec.

Driver:  python benchmarks/bench_rl_warehouse.py
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
ASSETS = os.path.join(WH, "assets")

# Objects that live on the warehouse floor (z=0). group=3 floor = collision only
# (the visible floor is the warehouse GLB; MJWarp/MuJoCo render their own plane).
OBJ_BODIES = """
    <body name="o0" pos="-0.8 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body name="o1" pos="-0.2 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="gold"/></body>
    <body name="o2" pos="0.4 0 0.30"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="copper"/></body>
    <body name="o3" pos="1.0 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="blue"/></body>
"""
MATS = """
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="copper" rgba="0.95 0.55 0.35 1" metallic="1.0" roughness="0.35"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
    <material name="floor"  rgba="0.45 0.46 0.50 1" metallic="0.0" roughness="0.6"/>
"""
CAM = '<camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>'

# For OURS: invisible collision floor (warehouse GLB is the visible one).
SCENE_WAREHOUSE = f"""
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>{MATS}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    {OBJ_BODIES}
    {CAM}
  </worldbody>
</mujoco>"""

# For FLAT renderers: a visible floor (they have no warehouse backdrop).
SCENE_FLAT = f"""
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>{MATS}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" material="floor"/>
    {OBJ_BODIES}
    {CAM}
  </worldbody>
</mujoco>"""


def make_cnn(h, w, dev):
    import torch, torch.nn as nn
    net = nn.Sequential(
        nn.Conv2d(3, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
    with torch.no_grad():
        n = net(torch.zeros(1, 3, h, w)).shape[1]
    head = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, 8))
    return nn.Sequential(net, head).to(dev)


def time_loop(one_step, steps, warmup):
    for _ in range(warmup):
        one_step()
    sim = ren = learn = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        s, r, l = one_step()
        sim += s; ren += r; learn += l
    return time.perf_counter() - t0, sim, ren, learn


def worker_ours(res, n, steps, warmup):
    import mujoco, torch
    from mujofil import WarpRenderer, RendererConfig
    m = mujoco.MjModel.from_xml_string(SCENE_WAREHOUSE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for d in datas: mujoco.mj_forward(m, d)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.6
    rr = WarpRenderer(cfg); rr.load_model(m)
    # full warehouse: IBL + GLB geometry + ceiling spotlights
    rr.load_ibl(os.path.join(ASSETS, "ibl", "warehouse_ibl_ibl.ktx"),
                os.path.join(ASSETS, "ibl", "warehouse_ibl_skybox.ktx"))
    rr.load_glb(os.path.join(ASSETS, "nvidia_floor.glb"))
    rr.load_glb(os.path.join(ASSETS, "nvidia_warehouse_tinted.glb"))
    rr.load_glb(os.path.join(ASSETS, "wall_patch.glb"))
    rr.set_ambient_intensity(9000.0)
    import json as _json
    with open(os.path.join(WH, "lamps.json")) as f:
        for x, y, z in _json.load(f):
            rr.add_spot_light(x, y, z, 0, 0, -1, 1.0, 0.97, 0.90, 32_000_000.0, 14.0, 55.0, 88.0, True)
    dev = "cuda"; net = make_cnn(res, res, dev); opt = torch.optim.Adam(net.parameters(), 1e-4)

    def one_step():
        t = time.perf_counter()
        for d in datas: mujoco.mj_step(m, d)
        t_sim = time.perf_counter() - t
        t = time.perf_counter()
        obs = rr.render_batch(m, datas, cam_id=0)
        x = obs[..., :3].permute(0, 3, 1, 2).float() / 255.0
        torch.cuda.synchronize(); t_ren = time.perf_counter() - t
        t = time.perf_counter()
        loss = net(x).pow(2).mean(); opt.zero_grad(True); loss.backward(); opt.step()
        torch.cuda.synchronize(); t_learn = time.perf_counter() - t
        return t_sim, t_ren, t_learn

    return time_loop(one_step, steps, warmup), n


def worker_mujoco(res, n, steps, warmup):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np, mujoco, torch
    m = mujoco.MjModel.from_xml_string(SCENE_FLAT)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for d in datas: mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=res, width=res)
    dev = "cuda"; net = make_cnn(res, res, dev); opt = torch.optim.Adam(net.parameters(), 1e-4)

    def one_step():
        t = time.perf_counter()
        for d in datas: mujoco.mj_step(m, d)
        t_sim = time.perf_counter() - t
        t = time.perf_counter()
        obs = np.empty((n, res, res, 3), np.uint8)
        for i, d in enumerate(datas):
            r.update_scene(d, camera=0); obs[i] = r.render()
        x = torch.from_numpy(obs).to(dev).permute(0, 3, 1, 2).float() / 255.0
        torch.cuda.synchronize(); t_ren = time.perf_counter() - t
        t = time.perf_counter()
        loss = net(x).pow(2).mean(); opt.zero_grad(True); loss.backward(); opt.step()
        torch.cuda.synchronize(); t_learn = time.perf_counter() - t
        return t_sim, t_ren, t_learn

    return time_loop(one_step, steps, warmup), n


def worker_mjwarp(res, n, steps, warmup):
    import mujoco, mujoco_warp as mjw, warp as wp, torch
    mjm = mujoco.MjModel.from_xml_string(SCENE_FLAT)
    m = mjw.put_model(mjm); d = mjw.make_data(mjm, nworld=n)
    rc = mjw.create_render_context(mjm, nworld=n, cam_res=(res, res), render_rgb=True,
                                   render_depth=False, use_textures=True, use_shadows=True)
    rgb = wp.zeros((n, res, res), dtype=wp.vec3)
    dev = "cuda"; net = make_cnn(res, res, dev); opt = torch.optim.Adam(net.parameters(), 1e-4)

    def one_step():
        t = time.perf_counter()
        mjw.step(m, d); wp.synchronize()
        t_sim = time.perf_counter() - t
        t = time.perf_counter()
        mjw.refit_bvh(m, d, rc); mjw.render(m, d, rc); mjw.get_rgb(rc, 0, rgb)
        x = wp.to_torch(rgb).permute(0, 3, 1, 2).float()
        torch.cuda.synchronize(); t_ren = time.perf_counter() - t
        t = time.perf_counter()
        loss = net(x).pow(2).mean(); opt.zero_grad(True); loss.backward(); opt.step()
        torch.cuda.synchronize(); t_learn = time.perf_counter() - t
        return t_sim, t_ren, t_learn

    return time_loop(one_step, steps, warmup), n


WORKERS = {"mujoco": worker_mujoco, "ours": worker_ours, "mjwarp": worker_mjwarp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(WORKERS), default=None)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    if args.worker:
        (wall, sim, ren, learn), n = WORKERS[args.worker](args.res, args.n, args.steps, args.warmup)
        print("RESULT " + json.dumps(dict(
            steps_per_s=n * args.steps / wall, sim=sim, render=ren, learn=learn)))
        return

    env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1", MUJOCO_GL="egl")

    def run(worker, res, n, steps=40, warmup=8):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", worker,
               "--res", str(res), "--n", str(n), "--steps", str(steps), "--warmup", str(warmup)]
        for _ in range(2):
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                return json.loads(line[0][7:])
        sys.stderr.write(f"[{worker} res={res} n={n}] failed: " +
                         "\n".join(r.stderr.splitlines()[-3:]) + "\n")
        return None

    print("\nWAREHOUSE parallel Vision-RL throughput: env-steps / second")
    print("(ours renders the FULL warehouse; flat renderers render objects only)\n")
    for res in [128, 256]:
        print(f"=== {res}x{res} ===")
        print(f"{'renderer':>14} {'N=8':>9} {'N=32':>9} {'N=64':>9}   (sim/ren/learn % @N=32)")
        print("-" * 76)
        for worker in ["mujoco", "ours", "mjwarp"]:
            sps = {}; mid = None
            for n in [8, 32, 64]:
                rd = run(worker, res, n)
                sps[n] = rd["steps_per_s"] if rd else float("nan")
                if n == 32 and rd:
                    tot = rd["sim"] + rd["render"] + rd["learn"]
                    mid = (100*rd["sim"]/tot, 100*rd["render"]/tot, 100*rd["learn"]/tot)
            br = f"{mid[0]:.0f}/{mid[1]:.0f}/{mid[2]:.0f}" if mid else "--"
            lbl = {"mujoco": "mujoco(flat)", "ours": "ours(warehouse PBR)", "mjwarp": "mjwarp(flat)"}[worker]
            print(f"{lbl:>14} {sps[8]:>9.0f} {sps[32]:>9.0f} {sps[64]:>9.0f}   {br}")
        print()
    print("ours = full warehouse (GLB meshes + IBL + 16 spotlights), photoreal PBR, zero-copy.")
    print("flat = bare objects only (cannot load the warehouse GLBs).")


if __name__ == "__main__":
    main()
