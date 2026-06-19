"""Aggressive render-path optimization probe. Renders a representative batched
scene and reports the render/flush/copy breakdown for a matrix of settings, so
we can attribute GPU cost and find redundancies WITHOUT guessing.

Each config runs in its OWN subprocess (one engine/process). The driver prints a
table. Worker is selected by env so we can toggle perFrame / flush_every too.

Run: DISPLAY=:1 python3 benchmarks/opt_probe.py
"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

SCENE = """<mujoco>
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
</mujoco>"""


def worker(res, n, steps, cfgkw):
    import numpy as np, mujoco
    from mujofil_warp import WarpRenderer, RendererConfig
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for d in datas:
        mujoco.mj_forward(m, d)
    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    cfg.enable_ssao = cfgkw.get("ssao", True)
    cfg.ssao_quality = cfgkw.get("ssao_quality", 3)
    cfg.ssao_ssct = cfgkw.get("ssct", True)
    cfg.enable_shadows = cfgkw.get("shadows", True)
    cfg.enable_msaa = cfgkw.get("msaa", True)
    cfg.msaa_samples = cfgkw.get("msaa_samples", 4)
    cfg.exposure = 1.4
    rr = WarpRenderer(cfg)
    rr.load_model(m)
    rr.add_directional_light(-0.3, 0.2, -1.0, 1.0, 0.98, 0.95, 60000.0, True)
    for _ in range(5):
        rr.render_batch(m, datas, cam_id=0)
    rr.reset_profile()
    t = time.perf_counter()
    for _ in range(steps):
        rr.render_batch(m, datas, cam_id=0)
    wall = time.perf_counter() - t
    p = rr.profile()
    nb = steps
    print("RESULT " + json.dumps(dict(
        cam_s=n * steps / wall,
        render_ms=p["render_ms"] / nb, flush_ms=p["flush_ms"] / nb,
        copy_ms=p["copy_ms"] / nb, wall_ms=wall / nb * 1000)))


def main():
    if os.environ.get("OPT_WORKER"):
        res = int(os.environ["OPT_RES"]); n = int(os.environ["OPT_N"])
        worker(res, n, int(os.environ["OPT_STEPS"]), json.loads(os.environ["OPT_CFG"]))
        return

    configs = [
        ("baseline ULTRA+SSCT", {}),
        ("ssct OFF",            {"ssct": False}),
        ("msaa OFF",            {"msaa": False}),
        ("ssao OFF",            {"ssao": False}),
        ("ssao+msaa OFF",       {"ssao": False, "msaa": False}),
        ("ssao+msaa+shadow OFF", {"ssao": False, "msaa": False, "shadows": False}),
    ]
    res, n, steps = 128, 256, 20
    print(f"\n{res}px N={n}  (per-batch ms)")
    print(f"{'config':22s} {'cam/s':>8s} {'render':>8s} {'flush':>8s} {'copy':>7s} {'wall':>8s}")
    print("-" * 70)
    for name, cfg in configs:
        env = dict(os.environ, OPT_WORKER="1", OPT_RES=str(res), OPT_N=str(n),
                   OPT_STEPS=str(steps), OPT_CFG=json.dumps({k: v for k, v in cfg.items() if k != "_env"}),
                   MUJOCO_GL="egl", MUJOFIL_NO_DRIVER_WARNING="1")
        env.update(cfg.get("_env", {}))
        r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=env, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"{name:22s}  FAILED  {r.stderr.strip().splitlines()[-1][:40] if r.stderr.strip() else ''}")
            continue
        d = json.loads(line[0][7:])
        print(f"{name:22s} {d['cam_s']:8.0f} {d['render_ms']:8.2f} {d['flush_ms']:8.2f} {d['copy_ms']:7.2f} {d['wall_ms']:8.2f}")


if __name__ == "__main__":
    main()
