"""Hero renders of photoreal GLB environments: a well-framed eye-level camera
sweep, rendered BOTH ways for comparison:
  GT  = render_batch over the gltfio GLB (full photoreal, in-engine tonemap)
  LAY = render_batch_layered over the ingested GLB (fast single instanced draw)
Saves a montage of N viewpoints for each, per scene.
"""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"; VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil_warp", "materials"))
sys.path.insert(0, HERE); sys.path.insert(0, VFM)
import numpy as np, torch, mujoco
from mujofil_warp import WarpRenderer, RendererConfig
from PIL import Image

# A free-floating camera body (freejoint) so we can pose an eye-level view that
# orbits the room centre and looks INWARD -- a proper "tour" of the environment.
RIG = """
<mujoco><option timestep="0.01"/><worldbody>
  <body name="rig" pos="0 0 1.4"><freejoint/>
    <geom type="box" size="0.05 0.05 0.05" rgba="0 0 0 0" contype="0" conaffinity="0"/>
    <camera name="cam" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>
  </body></worldbody></mujoco>"""


def ibl(name):
    base = os.path.join(HERE, "assets", "ibl")
    for cand in (name, name.replace("_ibl", ""), "studio"):
        d = os.path.join(base, cand)
        ii = os.path.join(d, f"{cand}_ibl.ktx"); ss = os.path.join(d, f"{cand}_skybox.ktx")
        if os.path.exists(ii):
            return ii, ss
    d = os.path.join(base, "studio")
    return os.path.join(d, "studio_ibl_ibl.ktx"), os.path.join(d, "studio_ibl_skybox.ktx")


def montage(a, res, cols, pad=6, bg=22):
    n = a.shape[0]; rows = int(np.ceil(n/cols))
    g = np.full((rows*res+(rows+1)*pad, cols*res+(cols+1)*pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols); y = pad+r*(res+pad); x = pad+c*(res+pad)
        g[y:y+res, x:x+res] = a[i]
    return g


def _look_quat(eye, target):
    """quaternion (w,x,y,z) for a +X-forward, +Z-up camera-rig body that makes the
    onboard camera (which looks down body -? per xyaxes) face `target`."""
    f = np.asarray(target) - np.asarray(eye); f = f/ (np.linalg.norm(f)+1e-9)
    up = np.array([0, 0, 1.0])
    s = np.cross(f, up); s = s/(np.linalg.norm(s)+1e-9)
    u = np.cross(s, f)
    # body +X = forward(f), +Y = left(-s? ) ; build rotation matrix cols = [f, ?, ?]
    # camera xyaxes="0 -1 0  0 0 1": cam x = body -Y, cam y = body +Z, cam -z = body +X
    # so body +X is the view direction -> set body X axis = f, Z = world up-ish.
    R = np.column_stack([f, np.cross(u, f), u])
    # matrix -> quat
    t = np.trace(R)
    if t > 0:
        sq = np.sqrt(t+1.0)*2; w = 0.25*sq
        x = (R[2,1]-R[1,2])/sq; y = (R[0,2]-R[2,0])/sq; z = (R[1,0]-R[0,1])/sq
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            sq = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w=(R[2,1]-R[1,2])/sq; x=0.25*sq; y=(R[0,1]+R[1,0])/sq; z=(R[0,2]+R[2,0])/sq
        elif i == 1:
            sq = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w=(R[0,2]-R[2,0])/sq; x=(R[0,1]+R[1,0])/sq; y=0.25*sq; z=(R[1,2]+R[2,1])/sq
        else:
            sq = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w=(R[1,0]-R[0,1])/sq; x=(R[0,2]+R[2,0])/sq; y=(R[1,2]+R[2,1])/sq; z=0.25*sq
    return np.array([w, x, y, z])


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "living"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 360
    from trailer.scenes import SCENES
    from glb_to_collision import _load_world_mesh, _xform_matrix
    s = SCENES[scene]; xform = [float(x) for x in s["xform"]]; glb = s["glb"]
    ambient = float(s.get("ambient", 8000.0))
    M = _xform_matrix(xform); vrt, _ = _load_world_mesh(glb, M)
    lo, hi = vrt.min(0), vrt.max(0); ctr = (lo+hi)/2.0; ext = hi-lo
    eye_h = lo[2] + 0.45*ext[2]          # ~mid-height eye line
    orbit = 0.34*float(np.linalg.norm(ext[:2]))   # inside the room

    m = mujoco.MjModel.from_xml_string(RIG)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    jadr = m.jnt_qposadr[0]
    datas = []
    for i in range(n):
        d = mujoco.MjData(m); ang = 2*np.pi*i/n
        eye = np.array([ctr[0]+orbit*np.cos(ang), ctr[1]+orbit*np.sin(ang), eye_h])
        q = _look_quat(eye, np.array([ctr[0], ctr[1], eye_h-0.1]))
        d.qpos[jadr:jadr+3] = eye; d.qpos[jadr+3:jadr+7] = q
        mujoco.mj_forward(m, d); datas.append(d)
    ii, ss = ibl(s.get("ibl", "studio_ibl"))
    out = os.path.join(HERE, "out", "hero"); os.makedirs(out, exist_ok=True)
    cols = 3 if n % 3 == 0 else (n if n < 4 else 2)

    # GT: full photoreal gltfio
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_shadows = True; cfg.exposure = 1.0
    r = WarpRenderer(cfg); r.load_model(m)
    if os.path.exists(ii): r.load_ibl(ii, ss)
    r.set_ambient_intensity(ambient); r.load_glb_xform(glb, xform)
    gt = r.render_batch(m, datas, cam_id=cam)[..., :3].clamp(0,255).byte().cpu().numpy()
    r.close()
    Image.fromarray(montage(gt, res, cols)).save(os.path.join(out, f"{scene}_photoreal.png"))

    # LAY: fast layered ingest
    cfgL = RendererConfig(); cfgL.width = cfgL.height = res; cfgL.batch_size = n
    cfgL.layered = True; cfgL.enable_shadows = True
    rL = WarpRenderer(cfgL); rL.load_model(m)
    if os.path.exists(ii): rL.load_ibl(ii, ss)
    rL.set_ambient_intensity(ambient*2.0); rL.load_glb_layered(glb, xform)
    lay = rL.render_batch_layered(m, datas, cam_id=cam, exposure=2.6)[..., :3].clamp(0,255).byte().cpu().numpy()
    rL.close()
    Image.fromarray(montage(lay, res, cols)).save(os.path.join(out, f"{scene}_layered.png"))
    print(f"saved {out}/{scene}_photoreal.png  and  {scene}_layered.png  (N={n} views)")


if __name__ == "__main__":
    main()
