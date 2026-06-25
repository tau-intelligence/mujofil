"""Egocentric (robot onboard camera) throughput: mujofil-warp vs vanilla MuJoCo
vs MJWarp, on the SAME robot/camera trajectory, batched into N worlds.

Each backend renders N egocentric views (one per world's robot camera):
  * mujoco  : stock mujoco.Renderer, EGL, N views rendered SEQUENTIALLY (no batch)
  * mjwarp  : MJWarp batched single-hit raycaster (flat Lambertian, NO PBR/textures
              for the GLB -- raycaster sees MJCF collision/visual geom only)
  * mujofil : mujofil-warp render_batch egocentric over the PHOTOREAL cafe GLB
              (full Filament PBR + IBL), one GPU sync for the whole batch

Subprocess per (backend, N) for clean Filament/CUDA/warp state. Metric = cameras/s.
NOTE fairness: only mujofil renders the photoreal GLB environment; mujoco & mjwarp
render the native robot + native room geom (GLB backdrops aren't supported by
either). So this measures egocentric *throughput*, with mujofil also doing the
heaviest visual work. A 'native room' variant keeps geometry identical for all 3.
"""
import os, sys, subprocess, time, json

HERE = "/home/mumuksh/mujofil-warp"
VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"

# Native room shared by all three backends (fair geometry); mujofil ALSO loads the
# cafe GLB on top when scene==glb.
ROOM = """
<mujoco><option timestep="0.01"/>
  <visual><global offwidth="4096" offheight="4096"/></visual>
  <asset>
    <material name="floor" rgba="0.55 0.42 0.30 1"/>
    <material name="wall"  rgba="0.82 0.80 0.76 1"/>
    <material name="red"   rgba="0.80 0.20 0.18 1"/>
    <material name="green" rgba="0.20 0.62 0.32 1"/>
    <material name="blue"  rgba="0.20 0.40 0.80 1"/>
    <material name="chrome" rgba="0.85 0.86 0.90 1" metallic="0.9" roughness="0.15"/>
  </asset>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.05" size="6 6 0.05" material="floor"/>
    <geom type="box" pos="6 0 2" size="0.05 6 2" material="wall"/>
    <geom type="box" pos="-6 0 2" size="0.05 6 2" material="wall"/>
    <geom type="box" pos="0 6 2" size="6 0.05 2" material="wall"/>
    <geom type="box" pos="0 -6 2" size="6 0.05 2" material="wall"/>
    <geom type="sphere" pos="0 0 1.0" size="0.6" material="chrome"/>
    <geom type="box" pos="4.6 0 0.5" size="0.5 0.7 0.5" material="red"/>
    <geom type="box" pos="-4.6 0 0.6" size="0.6 0.6 0.6" material="green"/>
    <geom type="cylinder" pos="0 4.6 0.7" size="0.5 0.7" material="blue"/>
    <light pos="0 0 3.8" dir="0 0 -1"/>
    <body name="robot" pos="2.6 0 0.35"><freejoint/>
      <geom type="box" size="0.3 0.2 0.15" rgba="0.12 0.12 0.14 1"/>
      <camera name="ego" pos="0.3 0 0.2" xyaxes="0 -1 0 0.18 0 0.98"/>
    </body>
  </worldbody></mujoco>"""


def _datas(m, n, mujoco):
    import numpy as np
    jadr = m.jnt_qposadr[0]
    out = []
    for i in range(n):
        d = mujoco.MjData(m); a = 2 * 3.14159265 * i / n
        d.qpos[jadr:jadr+3] = [2.6 * np.cos(a), 2.6 * np.sin(a), 0.35]
        h = a + 3.14159265
        d.qpos[jadr+3:jadr+7] = [np.cos(h/2), 0, 0, np.sin(h/2)]
        mujoco.mj_forward(m, d); out.append(d)
    return out


def worker_mujoco(res, iters, warmup, n):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np, mujoco, torch
    m = mujoco.MjModel.from_xml_string(ROOM)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    datas = _datas(m, n, mujoco)
    r = mujoco.Renderer(m, height=res, width=res)

    def step():
        out = np.empty((n, res, res, 3), np.uint8)
        for i, d in enumerate(datas):
            r.update_scene(d, camera=cam)
            out[i] = r.render()
        t = torch.from_numpy(out).to("cuda"); torch.cuda.synchronize(); return t
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return n * iters / (time.perf_counter() - t0)


def worker_mjwarp(res, iters, warmup, n):
    import mujoco, mujoco_warp as mjw, warp as wp, torch
    import numpy as np
    mjm = mujoco.MjModel.from_xml_string(ROOM)
    cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=n)
    # spread robots like _datas (write into the batched data's qpos)
    jadr = mjm.jnt_qposadr[0]
    qpos = d.qpos.numpy()
    for i in range(n):
        a = 2 * np.pi * i / n
        qpos[i, jadr:jadr+3] = [2.6 * np.cos(a), 2.6 * np.sin(a), 0.35]
        h = a + np.pi
        qpos[i, jadr+3:jadr+7] = [np.cos(h/2), 0, 0, np.sin(h/2)]
    d.qpos = wp.from_numpy(qpos, dtype=wp.float32)
    for _ in range(3):
        mjw.forward(m, d)
    wp.synchronize()
    rc = mjw.create_render_context(mjm, nworld=n, cam_res=(res, res), render_rgb=True,
                                   render_depth=False, use_textures=True, use_shadows=True)
    rgb = wp.zeros((n, res, res), dtype=wp.vec3)

    def step():
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        mjw.get_rgb(rc, cam, rgb)
        t = wp.to_torch(rgb); torch.cuda.synchronize(); return t
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return n * iters / (time.perf_counter() - t0)


def worker_mujofil(res, iters, warmup, n, use_glb):
    os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
    os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
    sys.path.insert(0, HERE); sys.path.insert(0, VFM)
    import mujoco, torch
    from mujofil import WarpRenderer, RendererConfig
    m = mujoco.MjModel.from_xml_string(ROOM)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    datas = _datas(m, n, mujoco)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_shadows = True; cfg.exposure = 1.0
    r = WarpRenderer(cfg); r.load_model(m)
    if use_glb:
        from trailer.scenes import SCENES
        s = SCENES["cafe"]
        d_ibl = os.path.join(HERE, "assets", "ibl", "studio")
        r.load_ibl(os.path.join(d_ibl, "studio_ibl_ibl.ktx"),
                   os.path.join(d_ibl, "studio_ibl_skybox.ktx"))
        r.set_ambient_intensity(float(s.get("ambient", 8000.0)))
        r.load_glb_xform(s["glb"], [float(x) for x in s["xform"]])

    def step():
        t = r.render_batch(m, datas, cam_id=cam)
        torch.cuda.synchronize(); return t
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return n * iters / (time.perf_counter() - t0)


def worker_mujofil_layered(res, iters, warmup, n, scene_key):
    """Full-PBR HIGH-FIDELITY path: GLB environment ingested into the instanced
    LAYERED renderables (albedo+normal+MR+emissive maps) -> N egocentric views in
    ONE instanced draw."""
    os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
    os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
    sys.path.insert(0, HERE); sys.path.insert(0, VFM)
    import mujoco, torch
    from mujofil import WarpRenderer, RendererConfig
    from trailer.scenes import SCENES
    m = mujoco.MjModel.from_xml_string(ROOM)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    datas = _datas(m, n, mujoco)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.layered = True; cfg.enable_shadows = True; cfg.exposure = 1.0
    r = WarpRenderer(cfg); r.load_model(m)
    s = SCENES[scene_key]
    d_ibl = os.path.join(HERE, "assets", "ibl", "studio")
    r.load_ibl(os.path.join(d_ibl, "studio_ibl_ibl.ktx"),
               os.path.join(d_ibl, "studio_ibl_skybox.ktx"))
    r.set_ambient_intensity(float(s.get("ambient", 8000.0)))
    r.load_glb_layered(s["glb"], [float(x) for x in s["xform"]])

    def step():
        t = r.render_batch_layered(m, datas, cam_id=cam, exposure=2.5)
        torch.cuda.synchronize(); return t
    for _ in range(warmup): step()
    t0 = time.perf_counter()
    for _ in range(iters): step()
    return n * iters / (time.perf_counter() - t0)


def run_child():
    which = sys.argv[2]; res = int(sys.argv[3]); n = int(sys.argv[4])
    iters, warmup = 8, 3
    if which == "mujoco":
        v = worker_mujoco(res, iters, warmup, n)
    elif which == "mjwarp":
        v = worker_mjwarp(res, iters, warmup, n)
    elif which == "mujofil_glb":
        v = worker_mujofil(res, iters, warmup, n, use_glb=True)
    elif which == "mujofil_layered":
        v = worker_mujofil_layered(res, iters, warmup, n, scene_key="sponza")
    elif which == "mujofil_native":
        v = worker_mujofil(res, iters, warmup, n, use_glb=False)
    else:
        v = -1
    print("RESULT " + json.dumps({"cps": v}))


def _child(which, res, n):
    cmd = (f"cd {HERE} && source .venv/bin/activate && "
           f"PYTHONPATH={HERE} MUJOFIL_WARP_BACKEND=gl MUJOCO_GL=egl "
           f"python benchmarks/ego_vs_baselines.py --child {which} {res} {n}")
    full = ["systemd-run", "--user", "-p", "LimitAS=infinity", "--quiet", "--wait",
            "--pipe", "--", "bash", "-c", cmd]
    try:
        out = subprocess.run(full, capture_output=True, text=True, timeout=600)
        for line in out.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[7:])["cps"]
        return None
    except Exception:
        return None


def main():
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    Ns = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["16", "64"])]
    backends = ["mujoco", "mjwarp", "mujofil_glb", "mujofil_layered"]
    label = {"mujoco": "vanilla MuJoCo (seq)", "mjwarp": "MJWarp (raycast,no-PBR)",
             "mujofil_native": "mujofil render_batch (native PBR)",
             "mujofil_glb": "mujofil render_batch (PHOTOREAL GLB)",
             "mujofil_layered": "mujofil LAYERED full-PBR GLB (1 draw)"}
    print(f"EGOCENTRIC throughput, res={res}  (cameras/sec, higher=better)\n")
    header = f"{'N':>5} | " + " | ".join(f"{label[b]:>38}" for b in backends)
    print(header); print("-" * len(header))
    for n in Ns:
        cells = []
        for b in backends:
            v = _child(b, res, n)
            cells.append("   n/a" if v is None else f"{v:>38.0f}".replace(label[b], ""))
            cells[-1] = ("   n/a" if v is None else f"{v:38.0f}")
        print(f"{n:>5} | " + " | ".join(cells))
    print("\nnote: mujofil_glb (render_batch) + mujofil_layered (single instanced draw)")
    print("both render a PHOTOREAL GLB w/ full PBR maps egocentrically; mujoco & mjwarp")
    print("render only the native room (neither can load a GLB). mujofil_layered=sponza")
    print("(102 normal+MR maps); mujofil_glb=cafe. Metric = total images/sec.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child()
    else:
        main()
