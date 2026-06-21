"""Visual tile-correctness check for the layered parallel-batch path.

For each environment it renders an N-world batch through render_batch_layered,
then saves a montage PNG (grid of all N tiles) so tile correctness can be
EYEBALLED, not just threshold-passed. It also prints per-tile diagnostics:
  * nonblack  : tiles that received content (every tile must be non-black)
  * uniq      : number of byte-distinct tiles (catches "tiles N..M identical to
                a neighbour" = routing/variation collapse even when non-black)
  * max_pair  : largest mean abs diff between well-separated tiles

Runs each env in its own subprocess (clean Filament/CUDA state).
"""
import os, sys, json, subprocess, itertools

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
_wm = os.path.join(HERE, "mujofil_warp", "materials")
if os.path.isdir(_wm):
    os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", _wm)

VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
SCN = os.path.join(VFM, "assets", "scenes")
ENV = os.path.join(VFM, "assets", "envmaps")
STUDIO = (os.path.join(ENV, "studio_ibl", "studio_ibl_ibl.ktx"),
          os.path.join(ENV, "studio_ibl", "studio_ibl_skybox.ktx"))
WH_IBL = (os.path.join(VFM, "warehouse/data/assets/ibl/warehouse_ibl_ibl.ktx"),
          os.path.join(VFM, "warehouse/data/assets/ibl/warehouse_ibl_skybox.ktx"))
OUT = os.path.join(HERE, "out", "tile_check")

CASES = [
    ("franka", "mjcf", os.path.join(VFM, "assets/models/franka_fr3_table.xml"), None),
    ("sponza", "glb", os.path.join(SCN, "sponza.glb"), STUDIO),
    ("living", "glb", os.path.join(SCN, "living.glb"), STUDIO),
    ("cafe", "glb", os.path.join(SCN, "cafe.glb"), STUDIO),
    ("gallery", "glb", os.path.join(SCN, "gallery.glb"), STUDIO),
    ("neon_street", "glb", os.path.join(SCN, "neon_street.glb"), STUDIO),
    ("scifi_dark", "glb", os.path.join(SCN, "scifi_dark.glb"), STUDIO),
    ("chess", "glb", os.path.join(SCN, "chess.glb"), STUDIO),
    ("nvidia_wh", "glb", os.path.join(VFM, "warehouse/data/assets/nvidia_warehouse_tinted.glb"), WH_IBL),
]

OBJ_XML = """
<mujoco><asset>
  <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.08"/>
  <material name="gold" rgba="1 0.78 0.34 1" metallic="1.0" roughness="0.25"/></asset>
<worldbody>
  <light pos="0 0 4" dir="0 0 -1"/>
  <body pos="-0.4 0 0.6"><freejoint/><geom type="sphere" size="0.3" material="chrome"/></body>
  <body pos="0.5 0 0.5"><freejoint/><geom type="box" size="0.25 0.25 0.25" material="gold"/></body>
  <body pos="0 0.5 0.45"><freejoint/><geom type="capsule" size="0.12 0.25" rgba="0.85 0.2 0.2 1"/></body>
  <camera name="c" pos="0 -3.2 1.3" xyaxes="1 0 0 0 0.36 0.93"/>
</worldbody></mujoco>"""


def worker(key, kind, target, ibl_i, ibl_s, n, res):
    import numpy as np, torch, mujoco
    from mujofil_warp import WarpRenderer, RendererConfig
    from PIL import Image

    if kind == "mjcf":
        m = mujoco.MjModel.from_xml_path(target)
    else:
        m = mujoco.MjModel.from_xml_string(OBJ_XML)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for i, d in enumerate(datas):
        ang = 6.2831853 * i / n
        dx, dy = 0.9 * float(np.cos(ang)), 0.9 * float(np.sin(ang))
        for j in range(m.njnt):
            t = m.jnt_type[j]
            adr = m.jnt_qposadr[j]
            if t == mujoco.mjtJoint.mjJNT_FREE:
                d.qpos[adr] += dx; d.qpos[adr + 1] += dy
            elif t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                d.qpos[adr] += 0.35 * float(np.sin(ang + j))
        mujoco.mj_forward(m, d)

    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.batch_size = n; cfg.layered = True
    r = WarpRenderer(cfg); r.load_model(m)
    if kind == "glb":
        r.load_ibl(ibl_i, ibl_s); r.load_glb(target); r.set_ambient_intensity(30000.0)
    cam = 0 if m.ncam > 0 else -1
    if cam < 0:
        r.set_free_camera(0, -3.2, 1.4, 0, 0, 0.4)
    imgs = r.render_batch_layered(m, datas, cam_id=cam)
    torch.cuda.synchronize()
    a = imgs[..., :3].clamp(0, 255).byte().cpu().numpy()   # (N,H,W,3)

    nonblack = int((a.reshape(n, -1).max(1) > 4).sum())
    uniq = len(set(a[i].tobytes() for i in range(n)))
    flat = a.reshape(n, -1).astype("float32")
    probe = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    mp = max(abs(flat[i] - flat[j]).mean()
             for i, j in itertools.combinations(probe, 2)) if n > 1 else 0.0

    # montage grid
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    H = W = res; pad = 2
    grid = np.zeros((rows * H + (rows - 1) * pad, cols * W + (cols - 1) * pad, 3), "uint8")
    for i in range(n):
        rr, cc = divmod(i, cols)
        grid[rr*(H+pad):rr*(H+pad)+H, cc*(W+pad):cc*(W+pad)+W] = a[i]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{key}_N{n}.png")
    Image.fromarray(grid).save(path)
    r.close()
    print("RESULT " + json.dumps(dict(key=key, n=n, nonblack=nonblack, uniq=uniq,
                                      max_pair=round(float(mp), 2), png=path)))
    sys.stdout.flush(); os._exit(0)


def main():
    if "--worker" in sys.argv:
        a = sys.argv; i = a.index("--worker")
        worker(a[i+1], a[i+2], a[i+3], a[i+4] or None, a[i+5] or None,
               int(a[i+6]), int(a[i+7]))
        return
    n, res = 16, 160
    print(f"=== TILE CHECK (N={n}, res={res}) -> {OUT} ===")
    n_fail = n_err = 0
    for key, kind, target, ibl in CASES:
        if not os.path.isfile(target):
            print(f"  SKIP {key} (missing)"); continue
        ii, ss = (ibl if ibl else ("", ""))
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               key, kind, target, ii, ss, str(n), str(res)]
        env = dict(os.environ, MUJOFIL_WARP_BACKEND="gl")
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), None)
        if not line:
            tail = (p.stderr.strip().splitlines() or ["<no output>"])[-1]
            print(f"  ERROR {key} :: {tail}"); n_err += 1; continue
        d = json.loads(line[len("RESULT "):])
        ok = d["nonblack"] == n and d["uniq"] == n
        if not ok:
            n_fail += 1
        flag = "" if ok else "  <-- CHECK (routing collapse: tiles not all distinct)"
        print(f"  {d['key']:<12} nonblack={d['nonblack']}/{n} uniq={d['uniq']}/{n} "
              f"max_pair={d['max_pair']:>6}  {os.path.basename(d['png'])}{flag}")
    print(f"=== {len(CASES)-n_fail-n_err} pass, {n_fail} fail, {n_err} error ===")
    # Real CI gate: nonzero exit if ANY environment's tiles collapsed, so this
    # class of bug (per-world routing failure on a new scene) can never ship
    # silently -- it fails the test suite for ANY environment, not just known ones.
    sys.exit(1 if (n_fail or n_err) else 0)


if __name__ == "__main__":
    main()
