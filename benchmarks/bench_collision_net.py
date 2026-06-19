"""Net full-pipeline throughput impact of the baked collision world.
Worker renders+steps ONE config (floor-only or full collision) and prints
env-steps/s. Driver runs both in separate processes (one engine per process).

Run: DISPLAY=:1 MUJOCO_GL=egl python3 bench_collision_net.py
"""
import os, sys, time, subprocess

MEN = os.path.expanduser("~/TAU-Tutorials/Drone_MARL_MuJoCo/MJ-drones-gym/mujoco_menagerie/franka_fr3")
COL = os.path.expanduser("~/Visual-Fidelity-Mujoco/assets/collision/gallery_collision.xml")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worker(which):
    sys.path.insert(0, HERE)
    import mujoco, torch, torch.nn as nn
    from mujofil_warp import WarpRenderer
    if which == "floor":
        inc = '<mujoco><include file="scene.xml"/><worldbody><geom type="plane" size="20 20 .1" group="3"/></worldbody></mujoco>'
    else:
        inc = f'<mujoco><include file="scene.xml"/><include file="{COL}"/></mujoco>'
    p = os.path.join(MEN, "_w.xml"); open(p, "w").write(inc)
    try:
        m = mujoco.MjModel.from_xml_path(p)
    finally:
        os.remove(p)
    N, RES = 256, 128
    ds = [mujoco.MjData(m) for _ in range(N)]
    for d in ds:
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    net = nn.Sequential(nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
    with torch.no_grad():
        nf = net(torch.zeros(1, 3, RES, RES)).shape[1]
    net = nn.Sequential(net, nn.Sequential(nn.Linear(nf, 512), nn.ReLU(), nn.Linear(512, 8))).to("cuda")
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    r = WarpRenderer(width=RES, height=RES, batch_size=N, preset="train")
    r.load_model(m)
    r.add_directional_light(-0.3, 0.2, -1, 1, 0.98, 0.95, 60000, True)

    def step():
        for d in ds:
            mujoco.mj_step(m, d)
        obs = r.render_batch(m, ds, cam_id=0)
        x = obs[..., :3].permute(0, 3, 1, 2).float() / 255.0
        loss = net(x).pow(2).mean(); opt.zero_grad(True); loss.backward(); opt.step()
        torch.cuda.synchronize()

    for _ in range(5):
        step()
    S = 25
    t = time.perf_counter()
    for _ in range(S):
        step()
    print(f"RESULT {N * S / (time.perf_counter() - t):.1f} {m.ngeom}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    if len(sys.argv) > 1:
        worker(sys.argv[1])
        return
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":1"),
               MUJOCO_GL="egl", MUJOFIL_NO_DRIVER_WARNING="1")
    res = {}
    for which in ["floor", "full"]:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), which],
                           env=env, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        res[which] = line[0].split()[1:] if line else ["nan", "?"]
    f = float(res["floor"][0]); c = float(res["full"][0])
    print(f"\nFull pipeline (physics+render train+CNN), 128px N=256, RTX4060:")
    print(f"  floor-only       : {f:7.0f} env-steps/s  (ngeom={res['floor'][1]})")
    print(f"  FULL collision   : {c:7.0f} env-steps/s  (ngeom={res['full'][1]})")
    print(f"  -> net impact    : {100*c/f:.1f}% of floor-only ({100*(1-c/f):.1f}% slower)")


if __name__ == "__main__":
    main()
