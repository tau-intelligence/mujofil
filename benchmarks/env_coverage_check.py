"""Verify the layered backdrop shows the FULL environment (not a corner crop) and
that per-world objects composite over it. Saves a hero + montage per scene with a
properly-framed eye-level camera derived from the scene's WORLD bounds.
"""
import os, sys, json
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/mumuksh/Visual-Fidelity-Mujoco")
sys.path.insert(0, "/home/mumuksh/Visual-Fidelity-Mujoco/scripts")
import numpy as np, torch, mujoco
import mujofil
from mujofil import WarpRenderer, RendererConfig
from PIL import Image
from trailer.scenes import SCENES
from glb_to_collision import _load_world_mesh, _xform_matrix

EN = "/home/mumuksh/Visual-Fidelity-Mujoco/assets/envmaps"
OUT = os.path.join(HERE, "out", "env_coverage")
os.makedirs(OUT, exist_ok=True)

OBJ = """<mujoco><asset>
 <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.08"/></asset>
<worldbody>
 <body pos="0 0 0.4"><freejoint/><geom type="sphere" size="0.22" material="chrome"/></body>
 <body pos="0.5 0 0.3"><freejoint/><geom type="box" size="0.18 0.18 0.18" rgba="1 .8 .3 1"/></body>
 <camera name="c" pos="0 -3 1.3" xyaxes="1 0 0 0 0.36 0.93"/></worldbody></mujoco>"""


def frame_camera(lo, hi):
    """Eye-level camera that frames the whole footprint, looking horizontally
    across the long axis from outside the shorter axis, at human eye height."""
    ctr = (lo + hi) / 2.0
    ext = hi - lo
    eye_h = lo[2] + min(1.6, 0.4 * ext[2] + 0.8)         # ~eye level
    look_h = lo[2] + 0.45 * ext[2]
    if ext[0] >= ext[1]:                                  # long in X -> stand in -Y
        back = 0.60 * ext[1] + 0.55 * ext[0]
        eye = (ctr[0], lo[1] - max(1.5, 0.15 * ext[1]) - 0.1 * ext[0] + (ctr[1]-lo[1]) - back + back, eye_h)
        eye = (ctr[0], ctr[1] - (0.5 * ext[1] + 0.45 * ext[0]), eye_h)
    else:                                                 # long in Y -> stand in -X
        eye = (ctr[0] - (0.5 * ext[0] + 0.45 * ext[1]), ctr[1], eye_h)
    return eye, (ctr[0], ctr[1], look_h)


def run(key, n, res):
    s = SCENES[key]
    xform = [float(x) for x in s["xform"]]
    ibl = s.get("ibl", "studio_ibl")
    ii = os.path.join(EN, ibl, f"{ibl}_ibl.ktx"); ss = os.path.join(EN, ibl, f"{ibl}_skybox.ktx")
    M = _xform_matrix(xform); v, _ = _load_world_mesh(s["glb"], M)
    lo, hi = v.min(0), v.max(0)
    # Use the scene's AUTHORED interior viewpoint (cam_robot = [ex,ey,ez,tx,ty,tz])
    # if present -- it's designed to frame the environment well from inside.
    # Pull the eye back slightly along the view dir for wider coverage.
    cr = s.get("cam_robot")
    if cr and len(cr) == 6:
        ex, ey, ez, tx, ty, tz = [float(x) for x in cr]
        vx, vy = ex - tx, ey - ty
        eye = (tx + vx * 1.8, ty + vy * 1.8, ez + 0.3)
        look = (tx, ty, tz)
    else:
        ctr = (lo + hi) / 2.0; ext = hi - lo
        eye = (ctr[0], ctr[1] - (0.5 * ext[1] + 0.4 * ext[0]), lo[2] + 1.5)
        look = (ctr[0], ctr[1], lo[2] + 0.4 * ext[2])

    m = mujoco.MjModel.from_xml_string(OBJ)
    datas = [mujoco.MjData(m) for _ in range(n)]
    for i, d in enumerate(datas):
        ang = 6.2831853 * i / n
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                adr = m.jnt_qposadr[j]
                d.qpos[adr] = look[0] + 0.5 * np.cos(ang)
                d.qpos[adr + 1] = look[1] + 0.5 * np.sin(ang)
                d.qpos[adr + 2] = look[2]
        mujoco.mj_forward(m, d)

    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.batch_size = n; cfg.layered = True
    r = WarpRenderer(cfg); r.load_model(m)
    if os.path.exists(ii): r.load_ibl(ii, ss)
    r.set_ambient_intensity(float(s.get("ambient", 9000.0)))
    r.load_glb_xform(s["glb"], xform)
    r.set_free_camera(eye[0], eye[1], eye[2], look[0], look[1], look[2])
    imgs = r.render_batch_layered(m, datas, cam_id=-1)
    torch.cuda.synchronize()
    a = imgs[..., :3].clamp(0, 255).byte().cpu().numpy()
    # coverage proxy: fraction of the frame that is non-background (got geometry)
    nonblack = int((a.reshape(n, -1).max(1) > 4).sum())
    uniq = len(set(a[i].tobytes() for i in range(n)))
    # how "full" the env is: % of pixels that aren't the clear/sky colour (rough)
    cov = float((a[0].reshape(-1, 3).max(1) > 8).mean()) * 100.0
    Image.fromarray(a[0]).save(os.path.join(OUT, f"{key}_hero.png"))
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols)); pad = 3
    grid = np.zeros((rows*res + (rows-1)*pad, cols*res + (cols-1)*pad, 3), "uint8")
    for i in range(n):
        rr, cc = divmod(i, cols); grid[rr*(res+pad):rr*(res+pad)+res, cc*(res+pad):cc*(res+pad)+res] = a[i]
    Image.fromarray(grid).save(os.path.join(OUT, f"{key}_montage.png"))
    r.close()
    print(f"RESULT {json.dumps(dict(key=key, n=n, nonblack=nonblack, uniq=uniq, coverage_pct=round(cov,1), bright=round(float(a.mean()),1)))}")


if __name__ == "__main__":
    key = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    run(key, n, res)
    sys.stdout.flush(); os._exit(0)
