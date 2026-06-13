"""Head-to-head: original mujofil (OpenGL batched, CPU readback) vs mujofil-warp
(Vulkan zero-copy), BOTH rendering the WAREHOUSE to a torch.cuda tensor.

Both render N envs of the same warehouse scene (GLB floor/walls/racks + IBL + 16
ceiling spotlights) and deliver the batch as a torch.cuda tensor — the endpoint a
learner consumes. The ONLY difference is the pixel path:

  mujofil_gl  : OpenGL batched render -> render_batch_rgb -> numpy (CPU) ->
                torch.from_numpy().cuda()   [the original optimized pipeline]
  warp_vk     : Vulkan render -> exportable buffer -> CUDA (zero-copy) -> torch
                [mujofil-warp]

cameras/sec (= N * iters / wall). Each in its own subprocess.
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

MATS = """
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
"""
OBJS = """
    <body name="o0" pos="-0.6 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body name="o1" pos="0.1 0 0.30"><freejoint/><geom type="sphere" size="0.24" material="gold"/></body>
    <body name="o2" pos="0.8 0 0.30"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="blue"/></body>
"""
SCENE = f"""
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>{MATS}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    {OBJS}
    <camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>
  </worldbody>
</mujoco>"""

IBL = (os.path.join(ASSETS, "ibl", "warehouse_ibl_ibl.ktx"),
       os.path.join(ASSETS, "ibl", "warehouse_ibl_skybox.ktx"))
GLBS = ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]


def _lamps():
    return json.load(open(os.path.join(WH, "lamps.json")))


def worker_gl(res, n, iters, warmup):
    """Original mujofil: OpenGL batched render -> CPU numpy -> torch.cuda."""
    import mujoco, torch
    from mujofil import native as vf
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for d in datas: mujoco.mj_forward(m, d)
    dptrs = [d._address for d in datas]
    c = vf.RendererConfig(); c.width = c.height = res
    c.use_vulkan = False          # OpenGL backend (the optimized readback path)
    c.enable_ssao = True; c.enable_shadows = True; c.enable_msaa = True
    c.exposure = 1.6; c.vsync = False
    r = vf.Renderer(c); r.initialize()
    b = vf.SceneBridge(r); b.load_model(m._address)
    b.load_ibl(*IBL)
    for g in GLBS: b.load_glb(os.path.join(ASSETS, g))
    b.set_ambient_intensity(9000.0)
    for x, y, z in _lamps():
        b.add_spot_light(x, y, z, 0, 0, -1, 1, 0.97, 0.90, 32_000_000.0, 14, 55, 88)

    def step():
        rgb = b.render_batch_rgb(m._address, dptrs, 0, res, res)  # (N,H,W,3) numpy CPU
        x = torch.from_numpy(rgb).to("cuda").float()              # upload CPU->GPU
        torch.cuda.synchronize()
        return x

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return n * iters / (time.perf_counter() - t0)


def worker_warp(res, n, iters, warmup):
    """mujofil-warp: Vulkan render -> CUDA zero-copy -> torch."""
    import mujoco, torch
    from mujofil_warp import WarpRenderer, RendererConfig
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for d in datas: mujoco.mj_forward(m, d)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.6
    r = WarpRenderer(cfg); r.load_model(m)
    r.load_ibl(*IBL)
    for g in GLBS: r.load_glb(os.path.join(ASSETS, g))
    r.set_ambient_intensity(9000.0)
    for x, y, z in _lamps():
        r.add_spot_light(x, y, z, 0, 0, -1, 1, 0.97, 0.90, 32_000_000.0, 14, 55, 88, True)

    def step():
        obs = r.render_batch(m, datas, cam_id=0)   # (N,H,W,4) cuda zero-copy
        x = obs[..., :3].float()
        torch.cuda.synchronize()
        return x

    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return n * iters / (time.perf_counter() - t0)


WORKERS = {"gl": worker_gl, "warp": worker_warp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=list(WORKERS), default=None)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    if args.worker:
        fps = WORKERS[args.worker](args.res, args.n, args.iters, args.warmup)
        print("FPS " + json.dumps({"fps": fps}))
        return

    env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1")

    def run(worker, res, n, iters=40, warmup=8):
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", worker,
               "--res", str(res), "--n", str(n), "--iters", str(iters), "--warmup", str(warmup)]
        for _ in range(2):
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith("FPS ")]
            if line:
                return json.loads(line[0][4:])["fps"]
        sys.stderr.write(f"[{worker} {res} N={n}] failed: " + "\n".join(r.stderr.splitlines()[-3:]) + "\n")
        return float("nan")

    print("\nWAREHOUSE render throughput -> torch.cuda (cameras/sec)")
    print("mujofil_gl = OpenGL batch + CPU readback + upload | warp = Vulkan zero-copy\n")
    for res in [128, 256, 512]:
        print(f"=== {res}x{res} ===")
        print(f"{'N':>5} {'mujofil_gl':>12} {'warp_zerocopy':>14} {'warp/gl':>9}")
        print("-" * 44)
        for n in [8, 32, 64]:
            gl = run("gl", res, n)
            wp_ = run("warp", res, n)
            sp = wp_ / gl if gl == gl and gl else float("nan")
            print(f"{n:>5} {gl:>12.0f} {wp_:>14.0f} {sp:>8.2f}x")
        print()


if __name__ == "__main__":
    main()
