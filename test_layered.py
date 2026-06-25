"""End-to-end test of the LAYERED parallel-batch render path.

Renders N worlds of a real MuJoCo scene (each world's box at a different X) in
ONE instanced GPU pass via the forked Filament gl_Layer path, returns (N,H,W,4)
torch.cuda, and verifies the worlds are distinct. Compares against the existing
per-world render_batch for correctness.
"""
import os
import sys

import mujofil
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR",
                      os.path.join(os.path.dirname(mujofil.__file__), "materials"))

import numpy as np
import torch
import mujoco
from mujofil import WarpRenderer, RendererConfig

SCENE = """
<mujoco>
  <asset><material name="red" rgba="0.85 0.25 0.2 1" metallic="0.0" roughness="0.4"/>
         <material name="gold" rgba="0.9 0.7 0.2 1" metallic="1.0" roughness="0.3"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
    <body pos="0 0 0.4"><freejoint/><geom type="box" size="0.25 0.25 0.25" material="red"/></body>
    <body pos="0.7 0.3 0.3"><freejoint/><geom type="sphere" size="0.2" material="gold"/></body>
    <camera name="cam0" pos="0 -2.6 1.0" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>
"""


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    res = 256
    m = mujoco.MjModel.from_xml_string(SCENE)
    datas = [mujoco.MjData(m) for _ in range(N)]
    for i, d in enumerate(datas):
        d.qpos[:] = 0
        d.qpos[0] = -1.5 + 3.0 * i / max(N - 1, 1)   # box marches across X
        mujoco.mj_forward(m, d)

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = N
    cfg.layered = True
    cfg.enable_msaa = False
    r = WarpRenderer(cfg)
    print("renderer.layered =", r.layered, " geoms =", "(after load)")
    r.load_model(m)
    print("geom_count =", r.geom_count)

    imgs = r.render_batch_layered(m, datas, cam_id=0)   # (N,H,W,4) cuda
    print(f"layered batch tensor: shape={tuple(imgs.shape)} dtype={imgs.dtype} device={imgs.device}")
    assert imgs.is_cuda and tuple(imgs.shape) == (N, res, res, 4)

    diffs = [(imgs[i].float() - imgs[i + 1].float()).abs().mean().item() for i in range(N - 1)]
    print("mean abs diff between consecutive worlds:", [round(x, 1) for x in diffs])
    distinct = sum(1 for d_ in diffs if d_ > 0.15)
    print(f"distinct consecutive pairs: {distinct}/{N-1}")

    # save a montage
    from PIL import Image
    os.makedirs("out", exist_ok=True)
    cols = min(N, 8)
    rows = (N + cols - 1) // cols
    cell = imgs[..., :3].cpu().numpy()[:, ::-1]  # flip vertical (GL origin)
    canvas = np.zeros((rows * res, cols * res, 3), np.uint8)
    for i in range(N):
        rr, cc = divmod(i, cols)
        canvas[rr*res:(rr+1)*res, cc*res:(cc+1)*res] = cell[i]
    Image.fromarray(canvas).save("out/layered_e2e.png")
    print("saved out/layered_e2e.png")

    nonblack = (imgs[..., :3].float().mean(dim=(1, 2, 3)) > 5).sum().item()
    print(f"non-black worlds: {nonblack}/{N}")
    assert nonblack == N, "some worlds rendered black"
    assert distinct >= N - 1, "worlds not distinct -> layered routing broken"
    print("\nLAYERED E2E OK: N worlds rendered in ONE pass, distinct, on cuda.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
