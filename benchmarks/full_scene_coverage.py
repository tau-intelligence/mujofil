"""Full-environment coverage render through the LAYERED parallel-batch pipeline.

Unlike tile_check (tiny props over a corner), this places each FULL scene exactly
as the product does -- load_glb_xform(scene xform) so the floor lands at z~0 in
world -- lights it with the scene's OWN IBL, and frames it with a WIDE camera
derived from the scene's world AABB so the whole environment is in view. It then
renders an N-world parallel batch and:
  * saves a montage of all N tiles (each = a full-environment frame), and
  * saves tile 0 as a standalone hero frame,
so large-coverage correctness AND per-tile routing can both be eyeballed.

Run (uncapped AS for CUDA):
  systemd-run --user -p LimitAS=infinity --quiet --wait --pipe -- bash -c \
    'cd ~/mujofil-warp && source .venv/bin/activate && \
     MUJOFIL_BACKEND=gl python benchmarks/full_scene_coverage.py'
"""
import os, sys, json, subprocess, itertools

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
_wm = os.path.join(HERE, "mujofil", "materials")
if os.path.isdir(_wm):
    os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", _wm)

VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
ENV = os.path.join(VFM, "assets", "envmaps")
OUT = os.path.join(HERE, "out", "full_coverage")

# scenes to render at full coverage (key from trailer.scenes)
SCENES = ["cafe", "gallery", "sponza", "neon_street", "living", "scifi_club"]


def _ibl(name):
    d = os.path.join(ENV, name)
    return (os.path.join(d, f"{name}_ibl.ktx"), os.path.join(d, f"{name}_skybox.ktx"))


# Per-world objects placed at the scene's robot spot, varied per world so tiling
# is visible even in full-coverage frames.
def _obj_xml():
    return """
<mujoco><asset>
  <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.08"/>
  <material name="gold" rgba="1 0.78 0.34 1" metallic="1.0" roughness="0.25"/></asset>
<worldbody>
  <body pos="0 0 0.35"><freejoint/><geom type="sphere" size="0.18" material="chrome"/></body>
  <body pos="0.45 0 0.30"><freejoint/><geom type="box" size="0.16 0.16 0.16" material="gold"/></body>
  <camera name="c" pos="0 -3 1.3" xyaxes="1 0 0 0 0.36 0.93"/>
</worldbody></mujoco>"""


def worker(key, n, res):
    import numpy as np, torch, mujoco
    from mujofil import WarpRenderer, RendererConfig
    from PIL import Image
    sys.path.insert(0, VFM)
    sys.path.insert(0, os.path.join(VFM, "scripts"))
    from trailer.scenes import SCENES as SC
    from glb_to_collision import _load_world_mesh, _xform_matrix

    s = SC[key]
    glb = s["glb"]
    xform = [float(x) for x in s["xform"]]
    ibl_name = s.get("ibl", "studio_ibl")
    ambient = float(s.get("ambient", 9000.0))

    # TRUE world-space bounds of the placed scene (same xform the renderer uses),
    # so the camera frames the WHOLE environment regardless of where it sits.
    M = _xform_matrix(xform)
    vrt, _ = _load_world_mesh(glb, M)
    lo, hi = vrt.min(0), vrt.max(0)
    ctr = (lo + hi) / 2.0
    ext = hi - lo
    diag = float(np.linalg.norm(ext[:2]))      # horizontal footprint

    m = mujoco.MjModel.from_xml_string(_obj_xml())
    datas = [mujoco.MjData(m) for _ in range(n)]
    for i, d in enumerate(datas):
        ang = 6.2831853 * i / n
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                adr = m.jnt_qposadr[j]
                d.qpos[adr] += 0.35 * float(np.cos(ang))
                d.qpos[adr + 1] += 0.35 * float(np.sin(ang))
        mujoco.mj_forward(m, d)

    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.batch_size = n; cfg.layered = True
    r = WarpRenderer(cfg); r.load_model(m)
    ii, ss = _ibl(ibl_name)
    if os.path.exists(ii):
        r.load_ibl(ii, ss)
    r.set_ambient_intensity(ambient)
    # place the FULL scene exactly as the product does (floor -> z~0 in world)
    r.load_glb_xform(glb, xform)
    # WIDE coverage camera. Back off along whichever horizontal axis is SHORTER
    # (so we look across the long dimension and frame the whole footprint), at a
    # gentle downward tilt, aiming at the scene centre a little above the floor.
    sxh, syh = float(ext[0]), float(ext[1])
    if sxh >= syh:
        # scene is long in X -> stand off in -Y, look across X
        back_ax = np.array([0.0, -1.0, 0.0]); span = sxh
    else:
        # scene is long in Y -> stand off in -X, look across Y
        back_ax = np.array([-1.0, 0.0, 0.0]); span = syh
    back = 0.62 * span + 2.5
    up = 0.30 * span + 2.0
    eye = np.array([ctr[0], ctr[1], lo[2]]) + back_ax * back + np.array([0, 0, up])
    look = np.array([ctr[0], ctr[1], lo[2] + 0.30 * ext[2]])
    r.set_free_camera(float(eye[0]), float(eye[1]), float(eye[2]),
                      float(look[0]), float(look[1]), float(look[2]))

    imgs = r.render_batch_layered(m, datas, cam_id=-1)
    torch.cuda.synchronize()
    a = imgs[..., :3].clamp(0, 255).byte().cpu().numpy()

    nonblack = int((a.reshape(n, -1).max(1) > 4).sum())
    uniq = len(set(a[i].tobytes() for i in range(n)))
    bright = float(a.mean())

    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(a[0]).save(os.path.join(OUT, f"{key}_hero.png"))
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols)); pad = 3
    grid = np.zeros((rows*res + (rows-1)*pad, cols*res + (cols-1)*pad, 3), "uint8")
    for i in range(n):
        rr, cc = divmod(i, cols)
        grid[rr*(res+pad):rr*(res+pad)+res, cc*(res+pad):cc*(res+pad)+res] = a[i]
    Image.fromarray(grid).save(os.path.join(OUT, f"{key}_montage_N{n}.png"))
    r.close()
    print("RESULT " + json.dumps(dict(key=key, n=n, nonblack=nonblack, uniq=uniq,
                                      bright=round(bright, 1))))
    sys.stdout.flush(); os._exit(0)


def main():
    if "--worker" in sys.argv:
        a = sys.argv; i = a.index("--worker")
        worker(a[i+1], int(a[i+2]), int(a[i+3])); return
    n = int(os.environ.get("COV_N", "9"))
    res = int(os.environ.get("COV_RES", "384"))
    print(f"=== FULL-SCENE COVERAGE (N={n}, res={res}) -> {OUT} ===")
    for key in SCENES:
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", key, str(n), str(res)]
        env = dict(os.environ, MUJOFIL_BACKEND="gl")
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), None)
        if not line:
            tail = (p.stderr.strip().splitlines() or ["<no output>"])[-1]
            print(f"  ERROR {key} :: {tail}"); continue
        d = json.loads(line[len("RESULT "):])
        flag = "" if (d["nonblack"] == n and d["uniq"] == n) else "  <-- CHECK"
        print(f"  {d['key']:<12} nonblack={d['nonblack']}/{n} uniq={d['uniq']}/{n} "
              f"bright={d['bright']}{flag}")


if __name__ == "__main__":
    main()
