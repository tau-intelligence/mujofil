"""Full physics+render+CNN pipeline throughput across resolutions, comparing
mujofil-warp (eval/train presets) to a vanilla-MuJoCo (EGL flat) baseline.

Also reports the WALLS variant: the same scene PLUS the collision proxy (floor +
4 bounding walls at group=3) that makes an imported environment actually walkable
-- so we can show the physics cost of "walkable, not just a backdrop".

env-steps/s = N * steps / wall. One step = physics(N) + render(N obs) + CNN fwd/bwd.
Each (worker,res) runs in its own subprocess. Run: DISPLAY=:1 python3 benchmarks/bench_res_sweep.py
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Bare scene: plane floor (robots stand) + 3 bodies. The plane alone already makes
# the floor solid; WALLS=1 adds 4 invisible bounding boxes (the collision proxy).
def scene_xml(walls: bool):
    wall_geoms = ""
    if walls:
        # 4 bounding walls at +/-X, +/-Y (group=3, invisible, contype/conaffinity 1)
        for tag, px, py, sx, sy in [("xlo", -10, 0, 0.2, 10), ("xhi", 10, 0, 0.2, 10),
                                    ("ylo", 0, -10, 10, 0.2), ("yhi", 0, 10, 10, 0.2)]:
            wall_geoms += (f'<geom type="box" pos="{px} {py} 2.5" size="{sx} {sy} 2.5" '
                           f'group="3" contype="1" conaffinity="1" rgba="0 0 0 0"/>')
    return f"""<mujoco><visual><global offwidth="1024" offheight="1024"/></visual>
  <asset><material name="c" rgba="0.9 0.9 0.95 1" metallic="1" roughness="0.1"/></asset>
  <worldbody>
    <geom type="plane" size="10 10 0.1"/>
    {wall_geoms}
    <body pos="-0.5 0 0.5"><freejoint/><geom type="sphere" size="0.25" material="c"/></body>
    <body pos="0.3 0 0.6"><freejoint/><geom type="box" size="0.2 0.2 0.2" material="c"/></body>
    <body pos="0.7 0.3 0.5"><freejoint/><geom type="sphere" size="0.22" material="c"/></body>
    <camera name="cam0" pos="0 -3 1.1" xyaxes="1 0 0 0 0.34 0.94"/>
  </worldbody></mujoco>"""


def make_cnn(res, dev):
    import torch, torch.nn as nn
    net = nn.Sequential(nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2),
                        nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
    with torch.no_grad():
        nf = net(torch.zeros(1, 3, res, res)).shape[1]
    return nn.Sequential(net, nn.Sequential(nn.Linear(nf, 512), nn.ReLU(),
                                            nn.Linear(512, 8))).to(dev)


def run_loop(step, steps, warm):
    for _ in range(warm):
        step()
    sim = ren = lr = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        a, b, c = step(); sim += a; ren += b; lr += c
    wall = time.perf_counter() - t0
    return wall, sim, ren, lr


def worker(kind, res, n, walls, steps, warm):
    import numpy as np, mujoco, torch
    m = mujoco.MjModel.from_xml_string(scene_xml(walls))
    ds = [mujoco.MjData(m) for _ in range(n)]
    for d in ds:
        mujoco.mj_forward(m, d)
    dev = "cuda"; net = make_cnn(res, dev)
    opt = torch.optim.Adam(net.parameters(), 1e-4)

    if kind == "mujoco":
        os.environ.setdefault("MUJOCO_GL", "egl")
        r = mujoco.Renderer(m, height=res, width=res)

        def step():
            t = time.perf_counter()
            for d in ds: mujoco.mj_step(m, d)
            ts = time.perf_counter() - t
            t = time.perf_counter()
            obs = np.empty((n, res, res, 3), np.uint8)
            for i, d in enumerate(ds):
                r.update_scene(d, camera=0); obs[i] = r.render()
            x = torch.from_numpy(obs).to(dev).permute(0, 3, 1, 2).float() / 255.0
            torch.cuda.synchronize(); tr = time.perf_counter() - t
            t = time.perf_counter()
            loss = net(x).pow(2).mean(); opt.zero_grad(True); loss.backward(); opt.step()
            torch.cuda.synchronize(); return ts, tr, time.perf_counter() - t
    else:
        from mujofil_warp import WarpRenderer
        preset = kind  # "eval" or "train"
        r = WarpRenderer(width=res, height=res, batch_size=n, preset=preset)
        r.load_model(m)
        r.add_directional_light(-0.3, 0.2, -1.0, 1.0, 0.98, 0.95, 60000.0, True)

        def step():
            t = time.perf_counter()
            for d in ds: mujoco.mj_step(m, d)
            ts = time.perf_counter() - t
            t = time.perf_counter()
            obs = r.render_batch(m, ds, cam_id=0)
            x = obs[..., :3].permute(0, 3, 1, 2).float() / 255.0
            torch.cuda.synchronize(); tr = time.perf_counter() - t
            t = time.perf_counter()
            loss = net(x).pow(2).mean(); opt.zero_grad(True); loss.backward(); opt.step()
            torch.cuda.synchronize(); return ts, tr, time.perf_counter() - t

    wall, sim, ren, lr = run_loop(step, steps, warm)
    print("RESULT " + json.dumps(dict(eps=n * steps / wall, sim=sim, ren=ren, lr=lr, steps=steps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker"); ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--walls", type=int, default=0)
    ap.add_argument("--steps", type=int, default=15); ap.add_argument("--warm", type=int, default=4)
    a = ap.parse_args()
    if a.worker:
        worker(a.worker, a.res, a.n, bool(a.walls), a.steps, a.warm); return

    base_env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1")
    base_env.pop("MUJOCO_GL", None)  # warp GL renderer must NOT see MUJOCO_GL=egl

    def run(kind, res, walls):
        env = dict(base_env)
        if kind == "mujoco":
            env["MUJOCO_GL"] = "egl"
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", kind,
               "--res", str(res), "--n", "256", "--walls", str(walls)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        ln = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        return json.loads(ln[0][7:]) if ln else None

    print("\nFull pipeline env-steps/s (physics+render+CNN), N=256, no walls")
    print(f"{'res':>6} {'mujoco(flat)':>14} {'eval(PBR)':>12} {'train(PBR)':>12}  {'train vs mjco':>13}")
    print("-" * 64)
    for res in [128, 256, 512]:
        mj = run("mujoco", res, 0); ev = run("eval", res, 0); tr = run("train", res, 0)
        mjr = mj["eps"] if mj else float("nan")
        evr = ev["eps"] if ev else float("nan")
        trr = tr["eps"] if tr else float("nan")
        print(f"{res:>6} {mjr:>14.0f} {evr:>12.0f} {trr:>12.0f}  {trr/mjr:>12.2f}x")

    print("\nWALKABLE cost: same scene WITH collision walls (floor+4 walls), train preset")
    print(f"{'res':>6} {'no walls':>12} {'with walls':>12}  {'overhead':>10}")
    print("-" * 46)
    for res in [128, 256]:
        a0 = run("train", res, 0); a1 = run("train", res, 1)
        r0 = a0["eps"] if a0 else float("nan"); r1 = a1["eps"] if a1 else float("nan")
        print(f"{res:>6} {r0:>12.0f} {r1:>12.0f}  {100*(1-r1/r0):>8.1f}%")


if __name__ == "__main__":
    main()
