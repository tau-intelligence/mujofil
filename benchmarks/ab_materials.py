"""A/B: render the SAME shared-camera layered batch and dump a montage. Run twice
(deployed vs reconstructed materials, swapped externally) and diff the PNGs to
prove the reconstructed M2 materials are pixel-identical in the shared-camera path
(i.e. instanced:true + the unused worldClip/usePerInstanceCam params changed
nothing for the published path)."""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
sys.path.insert(0, HERE)
import numpy as np, torch, mujoco
from mujofil_warp import WarpRenderer, RendererConfig
from PIL import Image

SCENE = """
<mujoco>
  <asset>
    <material name="red"   rgba="0.85 0.2 0.2 1" roughness="0.4" metallic="0.1"/>
    <material name="chrome" rgba="0.8 0.8 0.85 1" roughness="0.15" metallic="0.9"/>
    <material name="floor" rgba="0.5 0.5 0.55 1" roughness="0.8"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1" material="floor"/>
    <light pos="2 2 6" dir="-0.3 -0.3 -1"/>
    <body name="b0" pos="0 0 0.6"><freejoint/>
      <geom type="box" size="0.4 0.4 0.4" material="red"/></body>
    <body name="b1" pos="1.2 0 0.5"><freejoint/>
      <geom type="sphere" size="0.4" material="chrome"/></body>
  </worldbody>
</mujoco>"""


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "x"
    n = 6
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = []
    for i in range(n):
        d = mujoco.MjData(m)
        for b in range(2):
            adr = m.jnt_qposadr[b]
            ang = 2 * np.pi * (i / n) + b
            d.qpos[adr:adr+2] = [np.cos(ang) * (0.5 + b), np.sin(ang) * (0.5 + b)]
        mujoco.mj_forward(m, d)
        datas.append(d)
    cfg = RendererConfig(); cfg.width = cfg.height = 256; cfg.batch_size = n
    cfg.layered = True; cfg.enable_shadows = True
    r = WarpRenderer(cfg); r.load_model(m)
    imgs = r.render_batch_layered(m, datas)        # shared camera
    torch.cuda.synchronize()
    a = imgs[..., :3].clamp(0, 255).byte().cpu().numpy()
    res = a.shape[1]; cols = 3; rows = 2; pad = 2
    grid = np.zeros((rows*res+(rows-1)*pad, cols*res+(cols-1)*pad, 3), "uint8")
    for i in range(n):
        rr, cc = divmod(i, cols)
        grid[rr*(res+pad):rr*(res+pad)+res, cc*(res+pad):cc*(res+pad)+res] = a[i]
    out = os.path.join(HERE, "out", "ab_mat")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"ab_{tag}.npy"), a)
    Image.fromarray(grid).save(os.path.join(out, f"ab_{tag}.png"))
    print(f"[{tag}] saved out/ab_mat/ab_{tag}.png  uniq={len(set(a[i].tobytes() for i in range(n)))}/{n}")
    r.close()


if __name__ == "__main__":
    main()
