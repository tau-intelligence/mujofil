"""Egocentric (robot-mounted camera) ground truth via the EXISTING multi-RT batch
path. render_batch already syncs a PER-WORLD camera (sync_camera per env), so an
egocentric MJCF camera works TODAY -- each world renders from its own robot's
camera pose. This is the reference the layered (single-draw) version must match.

Native-geom scene (no GLB backdrop): a checkerboard floor + colored pillars +
a "robot" body carrying an onboard camera that looks forward. Each world rotates
the robot to a different heading -> each tile must show the pillars from that
heading (a genuinely different egocentric view per world).
"""
import os, sys, itertools
os.environ.setdefault("MUJOFIL_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)
import numpy as np, torch, mujoco
from mujofil import WarpRenderer, RendererConfig
from PIL import Image

# Native scene: floor, 4 distinct colored pillars around the origin, and a robot
# body with a forward-looking onboard camera.
SCENE = """
<mujoco>
  <asset>
    <material name="red"   rgba="0.85 0.2 0.2 1" roughness="0.5"/>
    <material name="green" rgba="0.2 0.8 0.3 1" roughness="0.5"/>
    <material name="blue"  rgba="0.2 0.4 0.9 1" roughness="0.5"/>
    <material name="yellow" rgba="0.9 0.8 0.2 1" roughness="0.5"/>
    <material name="floor" rgba="0.5 0.5 0.55 1" roughness="0.8"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="12 12 0.1" material="floor"/>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="px" type="box" pos="3 0 1" size="0.3 0.3 1" material="red"/>
    <geom name="nx" type="box" pos="-3 0 1" size="0.3 0.3 1" material="green"/>
    <geom name="py" type="box" pos="0 3 1" size="0.3 0.3 1" material="blue"/>
    <geom name="ny" type="box" pos="0 -3 1" size="0.3 0.3 1" material="yellow"/>
    <body name="robot" pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.2" rgba="0.9 0.9 0.95 1"/>
      <!-- onboard camera, looks along the body +X axis (forward) at eye height -->
      <camera name="ego" pos="0 0 0.3" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody>
</mujoco>"""


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    m = mujoco.MjModel.from_xml_string(SCENE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    assert cam >= 0
    # each world: rotate the robot to a different yaw so its onboard camera faces
    # a different pillar -> distinct egocentric view per world.
    datas = [mujoco.MjData(m) for _ in range(n)]
    for i, d in enumerate(datas):
        yaw = 2 * np.pi * i / n
        adr = m.jnt_qposadr[0]              # freejoint qpos: x y z qw qx qy qz
        d.qpos[adr:adr+3] = [0, 0, 0.5]
        d.qpos[adr+3:adr+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]  # yaw quaternion
        mujoco.mj_forward(m, d)

    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = n
    cfg.enable_shadows = True
    r = WarpRenderer(cfg); r.load_model(m)
    imgs = r.render_batch(m, datas, cam_id=cam)     # PER-WORLD egocentric camera
    torch.cuda.synchronize()
    a = imgs[..., :3].clamp(0, 255).byte().cpu().numpy()

    # which dominant pillar colour is centre-frame in each tile (proves the view
    # rotates with the robot).
    def centre_hue(t):
        cy, cx = t.shape[0]//2, t.shape[1]//2
        patch = t[cy-20:cy+20, cx-20:cx+20].reshape(-1, 3).mean(0)
        names = dict(red=(217,51,51), green=(51,204,76), blue=(51,102,229), yellow=(229,204,51))
        return min(names, key=lambda k: np.abs(patch - np.array(names[k])).sum())
    centres = [centre_hue(a[i]) for i in range(n)]
    out = os.path.join(HERE, "out", "ego_groundtruth")
    os.makedirs(out, exist_ok=True)
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n/cols)); pad = 3
    grid = np.zeros((rows*res+(rows-1)*pad, cols*res+(cols-1)*pad, 3), "uint8")
    for i in range(n):
        rr, cc = divmod(i, cols); grid[rr*(res+pad):rr*(res+pad)+res, cc*(res+pad):cc*(res+pad)+res] = a[i]
    Image.fromarray(grid).save(os.path.join(out, f"ego_render_batch_N{n}.png"))
    uniq = len(set(a[i].tobytes() for i in range(n)))
    probe = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    mp = max((a[i].astype("float32")-a[j]).__abs__().mean() for i,j in itertools.combinations(probe,2))
    print(f"render_batch egocentric: N={n} uniq={uniq}/{n} max_pair={mp:.1f} "
          f"centre_pillar_per_world={centres}")
    print(f"saved {out}/ego_render_batch_N{n}.png")
    r.close()


if __name__ == "__main__":
    main()
