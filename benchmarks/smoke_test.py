"""Ship-readiness smoke test: exercise every public render path + preset to catch
dtype/shape/crash regressions from the float16 RT + tonemap + GLB-ingest changes.
Asserts shapes, dtypes, value ranges, distinctness. Exits nonzero on any failure.
"""
import os, sys
os.environ.setdefault("MUJOFIL_WARP_BACKEND", "gl")
HERE = "/home/mumuksh/mujofil-warp"
os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", os.path.join(HERE, "mujofil", "materials"))
sys.path.insert(0, HERE)
import numpy as np, mujoco, torch
from mujofil import WarpRenderer, RendererConfig, make_config

SCENE = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.5 0.5 0.55 1"/>
    <light pos="0 0 4" dir="0 0 -1"/>
    <body name="b0" pos="0 0 0.5"><freejoint/>
      <geom type="box" size="0.3 0.3 0.3" rgba="0.85 0.3 0.2 1"/></body>
    <camera name="c" pos="0 -3 1.1" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>"""

fails = []


def check(name, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


def datas(m, n):
    out = []
    for i in range(n):
        d = mujoco.MjData(m); j = m.jnt_qposadr[0]
        ang = 2 * np.pi * i / n
        d.qpos[j:j+3] = [0.6*np.cos(ang), 0.6*np.sin(ang), 0.5]
        mujoco.mj_forward(m, d); out.append(d)
    return out


def main():
    m = mujoco.MjModel.from_xml_string(SCENE)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "c")

    print("== single render() ==")
    r = WarpRenderer(width=128, height=128)
    r.load_model(m); r.sync_camera(m, mujoco.MjData(m), cam_id=cam)
    img = r.render()
    check("render() shape (H,W,4)", tuple(img.shape) == (128, 128, 4))
    check("render() uint8", img.dtype == torch.uint8)
    check("render() cuda", img.is_cuda)
    check("render() nonblack", img[..., :3].float().mean().item() > 5)
    r.close()

    print("== render_batch (multi-RT) ==")
    n = 8; ds = datas(m, n)
    r = WarpRenderer(width=128, height=128, batch_size=n)
    r.load_model(m)
    out = r.render_batch(m, ds, cam_id=cam)
    check("render_batch shape (N,H,W,4)", tuple(out.shape) == (n, 128, 128, 4))
    check("render_batch uint8", out.dtype == torch.uint8)
    uniq = len(set(out[i].cpu().numpy().tobytes() for i in range(n)))
    check("render_batch tiles distinct", uniq == n)
    r.close()

    print("== render_batch preset=train ==")
    r = WarpRenderer(width=128, height=128, batch_size=n, preset="train")
    r.load_model(m)
    out = r.render_batch(m, ds, cam_id=cam)
    check("train shape", tuple(out.shape) == (n, 128, 128, 4))
    check("train nonblack", out[..., :3].float().mean().item() > 5)
    r.close()

    print("== render_batch preset=eval ==")
    r = WarpRenderer(width=128, height=128, batch_size=n, preset="eval")
    r.load_model(m)
    out = r.render_batch(m, ds, cam_id=cam)
    check("eval shape", tuple(out.shape) == (n, 128, 128, 4))
    r.close()

    print("== layered shared camera (float16 RT path) ==")
    r = WarpRenderer(width=128, height=128, batch_size=n, layered=True)
    r.load_model(m)
    out = r.render_batch_layered(m, ds, cam_id=-1)
    check("layered shared shape", tuple(out.shape) == (n, 128, 128, 4))
    check("layered shared uint8", out.dtype == torch.uint8)
    check("layered shared nonblack", out[..., :3].float().mean().item() > 5)
    r.close()

    print("== layered egocentric (per-world camera) ==")
    r = WarpRenderer(width=128, height=128, batch_size=n, layered=True)
    r.load_model(m)
    out = r.render_batch_layered(m, ds, cam_id=cam)
    check("layered ego shape", tuple(out.shape) == (n, 128, 128, 4))
    uniq = len(set(out[i].cpu().numpy().tobytes() for i in range(n)))
    check("layered ego tiles distinct", uniq == n)
    check("layered ego value range 0..255", int(out.max()) <= 255 and int(out.min()) >= 0)
    r.close()

    print("== reload safety (close + new model) ==")
    r = WarpRenderer(width=96, height=96, batch_size=4, layered=True)
    r.load_model(m)
    r.render_batch_layered(m, datas(m, 4), cam_id=cam)
    r.load_model(m)  # reload
    r.render_batch_layered(m, datas(m, 4), cam_id=-1)
    r.close()
    check("reload no crash", True)

    print()
    if fails:
        print(f"SMOKE FAILED: {len(fails)} -> {fails}")
        sys.exit(1)
    print("SMOKE PASSED: all render paths OK")


if __name__ == "__main__":
    main()
