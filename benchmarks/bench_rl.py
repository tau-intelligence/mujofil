"""Parallel Vision-RL throughput benchmark (NOT training to convergence).

One RL "step" = physics + render N observations + a representative CNN
forward+backward on the batched observations. We run a fixed number of steps and
report env-steps/sec (= N * steps / wall). This is the metric a vision-RL
practitioner actually cares about: how fast the sample loop turns.

Three renderers, each with its NATURAL physics pairing:
  - mujoco : CPU MuJoCo physics (N envs, sequential mj_step) + MuJoCo EGL render
             (flat) -> obs copied CPU->GPU -> CNN.
  - ours   : CPU MuJoCo physics (N envs, sequential mj_step) + our batched PBR
             render (zero-copy, obs already on GPU) -> CNN.
  - mjwarp : MJWarp GPU physics (batched N worlds) + MJWarp raycaster (flat,
             GPU-resident) -> CNN.

Each renderer runs in its own subprocess. We time sim / render / learn
separately so the breakdown is visible.

Driver:  python benchmarks/bench_rl.py
Worker:  python benchmarks/bench_rl.py --worker <name> --res <R> --n <N> --steps <K>
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
IBL_DIR = os.path.join(HERE, "assets", "ibl", "warehouse_new")

# Small free-floating scene with control inputs (a few actuated-ish bodies).
SCENE = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.1"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.25"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.3"/>
    <material name="floor"  rgba="0.45 0.46 0.50 1" metallic="0.0" roughness="0.6"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"/>
    <body pos="-0.5 0 0.5"><freejoint/><geom type="sphere" size="0.25" material="chrome"/></body>
    <body pos="0.2 0 0.6"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
    <body pos="0.7 0.3 0.5"><freejoint/><geom type="sphere" size="0.22" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.1" xyaxes="1 0 0 0 0.34 0.94"/>
  </worldbody>
</mujoco>
"""


def _ibl():
    return (os.path.join(IBL_DIR, "warehouse_new_ibl.ktx"),
            os.path.join(IBL_DIR, "warehouse_new_skybox.ktx"))


def make_cnn(h, w, dev):
    import torch.nn as nn
    import torch
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
    wall = time.perf_counter() - t0
    return wall, sim, ren, learn


# --------------------------------------------------------------------------- #
def worker_mujoco(res, n, steps, warmup):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np, mujoco, torch
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for d in datas: mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=res, width=res)
    dev = "cuda"; net = make_cnn(res, res, dev)
    opt = torch.optim.Adam(net.parameters(), 1e-4)

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


def worker_ours(res, n, steps, warmup):
    import numpy as np, mujoco, torch
    from mujofil_warp import WarpRenderer, RendererConfig
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for d in datas: mujoco.mj_forward(m, d)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.4
    rr = WarpRenderer(cfg); rr.load_model(m)
    ibl, sky = _ibl()
    if os.path.exists(ibl): rr.load_ibl(ibl, sky); rr.set_ambient_intensity(9000.0)
    rr.add_directional_light(-0.3, 0.2, -1.0, 1.0, 0.98, 0.95, 60000.0, True)
    dev = "cuda"; net = make_cnn(res, res, dev)
    opt = torch.optim.Adam(net.parameters(), 1e-4)

    def one_step():
        t = time.perf_counter()
        for d in datas: mujoco.mj_step(m, d)
        t_sim = time.perf_counter() - t
        t = time.perf_counter()
        obs = rr.render_batch(m, datas, cam_id=0)          # (n,H,W,4) cuda, zero-copy
        x = obs[..., :3].permute(0, 3, 1, 2).float() / 255.0
        torch.cuda.synchronize(); t_ren = time.perf_counter() - t
        t = time.perf_counter()
        loss = net(x).pow(2).mean(); opt.zero_grad(True); loss.backward(); opt.step()
        torch.cuda.synchronize(); t_learn = time.perf_counter() - t
        return t_sim, t_ren, t_learn

    return time_loop(one_step, steps, warmup), n


def worker_mjwarp(res, n, steps, warmup):
    import numpy as np, mujoco, mujoco_warp as mjw, warp as wp, torch
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    m = mjw.put_model(mjm); d = mjw.make_data(mjm, nworld=n)
    rc = mjw.create_render_context(mjm, nworld=n, cam_res=(res, res), render_rgb=True,
                                   render_depth=False, use_textures=True, use_shadows=True)
    rgb = wp.zeros((n, res, res), dtype=wp.vec3)
    dev = "cuda"; net = make_cnn(res, res, dev)
    opt = torch.optim.Adam(net.parameters(), 1e-4)

    def one_step():
        t = time.perf_counter()
        mjw.step(m, d); wp.synchronize()
        t_sim = time.perf_counter() - t
        t = time.perf_counter()
        mjw.refit_bvh(m, d, rc); mjw.render(m, d, rc); mjw.get_rgb(rc, 0, rgb)
        x = wp.to_torch(rgb).permute(0, 3, 1, 2).float()   # already 0..1 on GPU
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
        sps = n * args.steps / wall
        print("RESULT " + json.dumps(dict(
            steps_per_s=sps, sim=sim, render=ren, learn=learn, wall=wall, steps=args.steps)))
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
        sys.stderr.write(f"[{worker} res={res} n={n}] failed\n")
        return None

    print("\nParallel Vision-RL throughput: env-steps / second")
    print("(one step = physics + render N obs + CNN fwd/bwd; higher = better)\n")
    for res in [128, 256]:
        print(f"=== {res}x{res} ===")
        print(f"{'renderer':>12} {'N=8':>9} {'N=32':>9} {'N=64':>9}   (sim/ren/learn % at N=32)")
        print("-" * 74)
        for worker in ["mujoco", "ours", "mjwarp"]:
            sps = {}
            mid = None
            for n in [8, 32, 64]:
                res_d = run(worker, res, n)
                sps[n] = res_d["steps_per_s"] if res_d else float("nan")
                if n == 32 and res_d:
                    tot = res_d["sim"] + res_d["render"] + res_d["learn"]
                    mid = (100 * res_d["sim"] / tot, 100 * res_d["render"] / tot, 100 * res_d["learn"] / tot)
            br = f"{mid[0]:.0f}/{mid[1]:.0f}/{mid[2]:.0f}" if mid else "--"
            lbl = {"mujoco": "mujoco(flat)", "ours": "ours(PBR)", "mjwarp": "mjwarp(flat)"}[worker]
            print(f"{lbl:>12} {sps[8]:>9.0f} {sps[32]:>9.0f} {sps[64]:>9.0f}   {br}")
        print()
    print("mujoco/mjwarp = flat shading; ours = full PBR+IBL+shadows (zero-copy obs).")


if __name__ == "__main__":
    main()
