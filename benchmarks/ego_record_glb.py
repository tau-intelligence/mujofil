"""Egocentric robot-camera tour of a PHOTOREAL GLB environment.

This uses the MULTI-RT batch path (render_batch), which renders the FULL GLB
environment from each world's OWN onboard camera -- i.e. egocentric per-world
cameras DO work with photoreal GLB scenes. (The single-draw *layered* path can't
be egocentric over a shared GLB backdrop: its speedup comes from drawing the
backdrop ONCE for all tiles, which is only valid when every world shares a camera.
With per-world cameras the environment must be drawn per camera -- which is exactly
what render_batch does, batched into one GPU sync.)

N worlds share the photoreal living-room GLB; each world's robot drives a different
circular path inside the room, carrying a forward-facing onboard camera. Each tile
is that robot's eye view of the real environment. Stepped over time -> animated GIF.
"""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)
sys.path.insert(0, VFM)
sys.path.insert(0, os.path.join(VFM, "scripts"))
import numpy as np, torch, mujoco
from mujofil import WarpRenderer, RendererConfig
from PIL import Image

# A small wheeled robot with an onboard forward-facing camera. No floor/walls --
# the photoreal GLB environment IS the world; the robot just navigates inside it.
ROBOT = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="robot" pos="0 0 0.4">
      <freejoint/>
      <geom type="box" size="0.22 0.16 0.10" rgba="0.12 0.12 0.14 1"/>
      <geom type="sphere" pos="0.2 0 0.08" size="0.05" rgba="0.9 0.9 0.95 1"/>
      <!-- onboard camera: looks along body +X (forward), eye ~0.45m, slight tilt -->
      <camera name="ego" pos="0.22 0 0.16" xyaxes="0 -1 0 0.12 0 0.99"/>
    </body>
  </worldbody>
</mujoco>"""


def montage(a, res, pad=4, bg=16):
    n = a.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    g = np.full((rows * res + (rows + 1) * pad, cols * res + (cols + 1) * pad, 3), bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y = pad + r * (res + pad); x = pad + c * (res + pad)
        g[y:y + res, x:x + res] = a[i]
    return g


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "living"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    frames = int(sys.argv[4]) if len(sys.argv) > 4 else 72

    from trailer.scenes import SCENES
    s = SCENES[scene]
    glb = s["glb"]
    xform = [float(x) for x in s["xform"]]
    ibl_name = s.get("ibl", "studio_ibl")
    ambient = float(s.get("ambient", 8000.0))

    # Room centre on the floor plane (the xform translation is the GLB origin in
    # world; the living room is roughly centred there). Drive in a circle around it.
    cx, cy, cz = xform[12], xform[13], 0.0
    radius = 1.4
    speed = 1.3

    m = mujoco.MjModel.from_xml_string(ROBOT)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    jadr = m.jnt_qposadr[0]
    dt = m.opt.timestep
    substeps = 4

    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    speeds = speed * (0.8 + 0.4 * np.arange(n) / max(1, n - 1))
    datas = [mujoco.MjData(m) for _ in range(n)]

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    cfg.enable_shadows = True
    cfg.exposure = 1.0
    r = WarpRenderer(cfg)
    r.load_model(m)

    # IBL + the scene's own ambient, then place the photoreal GLB in world.
    def ibl_paths(name):
        base = os.path.join(HERE, "assets", "ibl")
        for cand in (name, name.replace("_ibl", ""), "studio"):
            d = os.path.join(base, cand)
            ii = os.path.join(d, f"{cand}_ibl.ktx"); ss = os.path.join(d, f"{cand}_skybox.ktx")
            if os.path.exists(ii):
                return ii, ss
        d = os.path.join(base, "studio")
        return os.path.join(d, "studio_ibl_ibl.ktx"), os.path.join(d, "studio_ibl_skybox.ktx")
    ii, ss = ibl_paths(ibl_name)
    if os.path.exists(ii):
        r.load_ibl(ii, ss)
    r.set_ambient_intensity(ambient)
    r.load_glb_xform(glb, xform)   # the FULL photoreal environment, native to the scene

    def place(d, t, i):
        ang = phases[i] + speeds[i] * t
        x, y = cx + radius * np.cos(ang), cy + radius * np.sin(ang)
        heading = ang + np.pi / 2.0                # face tangent (forward along path)
        qw, qz = np.cos(heading / 2.0), np.sin(heading / 2.0)
        d.qpos[jadr:jadr + 3] = [x, y, cz + 0.4]
        d.qpos[jadr + 3:jadr + 7] = [qw, 0, 0, qz]
        d.qvel[:] = 0.0

    out_dir = os.path.join(HERE, "out", "ego_record_glb")
    os.makedirs(out_dir, exist_ok=True)
    gif_frames = []
    t = 0.0
    for f in range(frames):
        for i, d in enumerate(datas):
            place(d, t, i)
            mujoco.mj_forward(m, d)
        imgs = r.render_batch(m, datas, cam_id=cam)   # egocentric over the GLB env
        torch.cuda.synchronize()
        a = imgs[..., :3].clamp(0, 255).byte().cpu().numpy()
        gif_frames.append(Image.fromarray(montage(a, res)))
        t += substeps * dt
        if f % 12 == 0:
            print(f"  frame {f+1}/{frames}")

    gif_path = os.path.join(out_dir, f"egocentric_glb_{scene}_N{n}.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=55, loop=0, optimize=True)
    gif_frames[len(gif_frames) // 3].save(
        os.path.join(out_dir, f"egocentric_glb_{scene}_N{n}_still.png"))
    print(f"saved {gif_path}  ({len(gif_frames)} frames, {n} egocentric tiles, scene={scene})")
    r.close()


if __name__ == "__main__":
    main()
