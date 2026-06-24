"""Record the TILED layered render with ACTIVE EGOCENTRIC cameras in motion.

N worlds share one model but each robot drives a different circular path; every
robot carries an onboard camera. As the physics steps, render_batch_layered
(cam_id=ego) renders all N egocentric views in ONE instanced draw per frame. We
montage the tiles each step and write an animated GIF so you can see each tile's
robot-eye view evolve as its world advances.
"""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil_warp", "materials"))
sys.path.insert(0, HERE)
import numpy as np, torch, mujoco
from mujofil_warp import WarpRenderer, RendererConfig
from PIL import Image

# A WELL-LIT enclosed native room: wood floor, light walls + ceiling, a few
# coloured furniture-like landmarks and a metallic centrepiece sculpture, plus a
# wheeled "robot" body (freejoint) carrying a forward-facing onboard camera.
# Everything is NATIVE MJCF geometry (no GLB backdrop) so the per-world egocentric
# cameras are fully correct in the single instanced layered draw. Lit with studio
# IBL + warm ceiling fill so each robot-eye tile looks like a real lit interior.
SCENE = """
<mujoco>
  <option timestep="0.01"/>
  <asset>
    <material name="floor"  rgba="0.55 0.42 0.30 1" roughness="0.5"  metallic="0.0"/>
    <material name="wall"   rgba="0.82 0.80 0.76 1" roughness="0.9"  metallic="0.0"/>
    <material name="ceil"   rgba="0.90 0.90 0.92 1" roughness="0.95" metallic="0.0"/>
    <material name="red"    rgba="0.80 0.20 0.18 1" roughness="0.45" metallic="0.05"/>
    <material name="green"  rgba="0.20 0.62 0.32 1" roughness="0.45" metallic="0.05"/>
    <material name="blue"   rgba="0.20 0.40 0.80 1" roughness="0.45" metallic="0.05"/>
    <material name="teal"   rgba="0.18 0.62 0.62 1" roughness="0.40" metallic="0.1"/>
    <material name="wood2"  rgba="0.45 0.30 0.20 1" roughness="0.55" metallic="0.0"/>
    <material name="gold"   rgba="0.90 0.74 0.36 1" roughness="0.18" metallic="0.9"/>
    <material name="chrome" rgba="0.85 0.86 0.90 1" roughness="0.12" metallic="0.95"/>
  </asset>
  <worldbody>
    <!-- room shell: floor, ceiling, 4 walls (half-size 6 -> 12x12, height 4) -->
    <geom name="floor" type="box" pos="0 0 -0.05" size="6 6 0.05" material="floor"/>
    <geom name="ceil"  type="box" pos="0 0 4.0"  size="6 6 0.05" material="ceil"/>
    <geom name="wpx" type="box" pos="6 0 2"  size="0.05 6 2" material="wall"/>
    <geom name="wnx" type="box" pos="-6 0 2" size="0.05 6 2" material="wall"/>
    <geom name="wpy" type="box" pos="0 6 2"  size="6 0.05 2" material="wall"/>
    <geom name="wny" type="box" pos="0 -6 2" size="6 0.05 2" material="wall"/>

    <!-- metallic centrepiece sculpture -->
    <geom name="pedestal" type="cylinder" pos="0 0 0.4" size="0.8 0.4" material="wood2"/>
    <geom name="orb"      type="sphere"   pos="0 0 1.2" size="0.6" material="chrome"/>
    <geom name="ring"     type="cylinder" pos="0 0 1.95" size="0.45 0.06" material="gold"/>

    <!-- coloured furniture-like landmarks around the room -->
    <geom name="lm_red"   type="box"      pos="4.6 0 0.5"  size="0.5 0.7 0.5" material="red"/>
    <geom name="lm_green" type="box"      pos="-4.6 0 0.6" size="0.6 0.6 0.6" material="green"/>
    <geom name="lm_blue"  type="cylinder" pos="0 4.6 0.7"  size="0.5 0.7"     material="blue"/>
    <geom name="lm_teal"  type="box"      pos="0 -4.6 0.45" size="0.7 0.5 0.45" material="teal"/>
    <geom name="lm_gold"  type="sphere"   pos="3.2 3.2 0.5" size="0.5"        material="gold"/>
    <geom name="lm_wood"  type="box"      pos="-3.2 -3.2 0.4" size="0.6 0.6 0.4" material="wood2"/>

    <!-- warm ceiling fill so the interior reads as well lit -->
    <light pos="0 0 3.8"  dir="0 0 -1" diffuse="1.0 0.95 0.85"/>
    <light pos="3 3 3.6"  dir="-0.4 -0.4 -1" diffuse="0.6 0.6 0.7"/>
    <light pos="-3 -3 3.6" dir="0.4 0.4 -1" diffuse="0.6 0.6 0.7"/>

    <body name="robot" pos="3 0 0.35">
      <freejoint/>
      <geom type="box" size="0.32 0.22 0.16" rgba="0.12 0.12 0.14 1"/>
      <geom type="sphere" pos="0.28 0 0.1" size="0.07" rgba="0.9 0.9 0.95 1"/>
      <!-- onboard camera: looks along the body +X (forward), slight upward tilt -->
      <camera name="ego" pos="0.30 0 0.20" xyaxes="0 -1 0 0.18 0 0.98"/>
    </body>
  </worldbody>
</mujoco>"""


def montage(a, res, pad=4, bg=18):
    n = a.shape[0]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    g = np.full((rows * res + (rows + 1) * pad, cols * res + (cols + 1) * pad, 3),
                bg, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y = pad + r * (res + pad)
        x = pad + c * (res + pad)
        g[y:y + res, x:x + res] = a[i]
    return g


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 224
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    substeps = 4          # physics steps per recorded frame
    radius = 2.6
    speed = 1.6           # base angular speed (rad/s of the drive circle)

    m = mujoco.MjModel.from_xml_string(SCENE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    assert cam >= 0
    jadr = m.jnt_qposadr[0]   # freejoint qpos: x y z qw qx qy qz

    # each world gets a different phase + slightly different speed so the tiles
    # show visibly different robot-eye views at any instant.
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    speeds = speed * (0.8 + 0.4 * np.arange(n) / max(1, n - 1))

    datas = [mujoco.MjData(m) for _ in range(n)]
    dt = m.opt.timestep

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    cfg.layered = True
    cfg.enable_shadows = True
    cfg.exposure = 1.0
    r = WarpRenderer(cfg)
    r.load_model(m)
    # Studio image-based lighting for soft, even interior illumination + nice
    # metallic reflections on the centrepiece. Skybox stays hidden behind the room
    # walls; the IBL irradiance is what makes each robot-eye tile read as well lit.
    ibl_dir = os.path.join(HERE, "assets", "ibl", "studio")
    try:
        r.load_ibl(os.path.join(ibl_dir, "studio_ibl_ibl.ktx"),
                   os.path.join(ibl_dir, "studio_ibl_skybox.ktx"))
    except Exception as e:
        print("  (IBL load skipped:", e, ")")
    try:
        r.set_ambient_intensity(38000.0)
    except Exception:
        pass

    def place(d, t, i):
        # drive the robot around a circle of `radius`, with the onboard camera
        # aimed INWARD at the central tower so each view frames the metallic tower
        # with the coloured landmark pillars sweeping behind it as the world spins.
        ang = phases[i] + speeds[i] * t
        x, y = radius * np.cos(ang), radius * np.sin(ang)
        heading = ang + np.pi                            # face the centre (inward)
        qw, qz = np.cos(heading / 2.0), np.sin(heading / 2.0)
        d.qpos[jadr:jadr + 3] = [x, y, 0.35]
        d.qpos[jadr + 3:jadr + 7] = [qw, 0, 0, qz]
        d.qvel[:] = 0.0

    out_dir = os.path.join(HERE, "out", "ego_record")
    os.makedirs(out_dir, exist_ok=True)
    gif_frames = []
    t = 0.0
    for f in range(frames):
        for i, d in enumerate(datas):
            place(d, t, i)
            mujoco.mj_forward(m, d)
        imgs = r.render_batch_layered(m, datas, cam_id=cam)   # tiled egocentric
        torch.cuda.synchronize()
        a = imgs[..., :3].clamp(0, 255).byte().cpu().numpy()
        gif_frames.append(Image.fromarray(montage(a, res)))
        t += substeps * dt
        if f % 10 == 0:
            print(f"  frame {f+1}/{frames}")

    gif_path = os.path.join(out_dir, f"egocentric_tiled_N{n}.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=50, loop=0, optimize=True)
    # also drop a representative still
    gif_frames[len(gif_frames) // 3].save(
        os.path.join(out_dir, f"egocentric_tiled_N{n}_still.png"))
    print(f"saved {gif_path}  ({len(gif_frames)} frames, {n} egocentric tiles)")
    r.close()


if __name__ == "__main__":
    main()
