"""Subprocess-isolated A/B: MJWarp flat raycaster vs Filament warehouse PBR.

The two renderers can't share one process (MJWarp's render teardown throws CUDA
700 after the Filament/warehouse render, and Filament's own teardown double-frees
on exit). So each render runs in its OWN subprocess and writes a PNG; the driver
then composites them into a single side-by-side image.

State is shared exactly: the FLAT worker steps physics, settles the objects, and
writes qpos.npy AFTER stepping; the PBR worker loads that exact qpos so both
images show the identical physical state.

Run:
    python scripts/ab_compare.py --res 1024 --steps 400 --camera floor
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _scene import build_xml  # noqa: E402

WAREHOUSE_DIR = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse"


# --------------------------------------------------------------------------- #
# Workers (run as `python ab_compare.py --worker {flat,pbr} ...`)
# --------------------------------------------------------------------------- #
def worker_flat(qpos_path, out_path, res, steps, seed):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    mjm = mujoco.MjModel.from_xml_string(build_xml())
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1)

    qpos = d.qpos.numpy()
    rng = np.random.default_rng(seed)
    qpos[:] += rng.uniform(-0.03, 0.03, size=qpos.shape)
    wp.copy(d.qpos, wp.array(qpos, dtype=float))

    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    for _ in range(steps):
        mjw.step(m, d)
    wp.synchronize()

    # save the settled state for the PBR worker
    np.save(qpos_path, d.qpos.numpy()[0])

    W = H = res
    rc = mjw.create_render_context(
        mjm, nworld=1, cam_res=(W, H), render_rgb=True, render_depth=False,
        use_textures=True, use_shadows=True)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    rgb = wp.zeros((1, H, W), dtype=wp.vec3)
    mjw.get_rgb(rc, 0, rgb)
    arr = rgb.numpy()[0]
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 + 1e-3 \
        else np.clip(arr, 0, 255).astype(np.uint8)

    from PIL import Image
    Image.fromarray(arr).save(out_path)
    print(f"FLAT_OK mean={arr.mean():.1f} std={arr.std():.1f}")


def worker_pbr(qpos_path, out_path, res, camera):
    import mujoco
    from mujofil import native as _vf
    sys.modules.setdefault("_vf_mujoco_native", _vf)
    sys.path.insert(0, WAREHOUSE_DIR)
    from warehouse_env import Warehouse

    qpos = np.load(qpos_path)
    mjm = mujoco.MjModel.from_xml_string(build_xml())
    host = mujoco.MjData(mjm)
    host.qpos[:] = qpos
    mujoco.mj_forward(mjm, host)

    W = H = res
    wh = Warehouse()
    cfg = wh.default_config(W, H)
    cfg.exposure = 1.6          # warehouse default 2.5 blows out the bright floor
    r = _vf.Renderer(cfg)
    r.initialize()
    b = _vf.SceneBridge(r)
    b.load_model(mjm._address)
    wh.load(b)
    # The warehouse ceiling spots have shadow-casting disabled, so objects look
    # ungrounded. Add one near-vertical directional light WITH shadows to cast a
    # contact shadow onto the floor without washing out the warehouse look.
    b.add_directional_light(-0.15, 0.12, -1.0, 1.0, 0.98, 0.95, 11000.0, True)
    b.sync_transforms(mjm._address, host._address)
    # Use the SAME camera as the flat render (scene cam0) for a fair A/B.
    b.sync_camera(mjm._address, host._address, 0)
    img = np.asarray(r.render())[:, :, :3]

    from PIL import Image
    Image.fromarray(img).save(out_path)
    print(f"PBR_OK mean={img.mean():.1f} std={img.std():.1f}")
    # NOTE: do not r.destroy() (teardown double-free); process exit is clean.


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def label(img, text):
    """Draw a caption bar at the top of an image array."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    bar_h = max(28, im.height // 22)
    d.rectangle([0, 0, im.width, bar_h], fill=(0, 0, 0))
    d.text((10, bar_h // 2 - 8), text, fill=(255, 255, 255))
    return np.asarray(im)


def run_worker(mode, **kw):
    args = [sys.executable, os.path.abspath(__file__), "--worker", mode]
    for k, v in kw.items():
        args += [f"--{k}", str(v)]
    env = dict(os.environ, MUJOFIL_NO_DRIVER_WARNING="1")
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    tag = [ln for ln in r.stdout.splitlines() if "_OK" in ln]
    ok = bool(tag)
    print(f"  [{mode}] {'ok: ' + tag[0] if ok else 'FAILED'}")
    if not ok:
        # surface the tail of stderr for debugging
        err = "\n".join(r.stderr.splitlines()[-6:])
        print(f"  [{mode}] stderr tail:\n{err}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["flat", "pbr"], default=None)
    ap.add_argument("--qpos", default="out/_ab_qpos.npy")
    ap.add_argument("--outpath", default=None)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="floor")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    if args.worker == "flat":
        worker_flat(args.qpos, args.outpath, args.res, args.steps, args.seed)
        return
    if args.worker == "pbr":
        worker_pbr(args.qpos, args.outpath, args.res, args.camera)
        return

    # --- driver ---
    os.makedirs(args.out, exist_ok=True)
    qpos = os.path.join(args.out, "_ab_qpos.npy")
    flat_png = os.path.join(args.out, "ab_flat.png")
    pbr_png = os.path.join(args.out, "ab_pbr.png")

    print("[1] FLAT worker (MJWarp physics + raycaster) ...")
    ok_flat = run_worker("flat", qpos=qpos, outpath=flat_png, res=args.res,
                         steps=args.steps, seed=args.seed)
    if not ok_flat:
        print("  flat worker failed; aborting.")
        return

    print("[2] PBR worker (Filament warehouse) ...")
    ok_pbr = run_worker("pbr", qpos=qpos, outpath=pbr_png, res=args.res,
                        camera=args.camera)
    if not ok_pbr:
        print("  pbr worker failed; aborting.")
        return

    from PIL import Image
    flat = np.asarray(Image.open(flat_png).convert("RGB"))
    pbr = np.asarray(Image.open(pbr_png).convert("RGB"))
    h = min(flat.shape[0], pbr.shape[0])
    flat = label(flat[:h], "MJWarp raycaster (flat)")
    pbr = label(pbr[:h], "Filament PBR (warehouse)")
    gap = np.full((h, 6, 3), 255, np.uint8)
    strip = np.concatenate([flat, gap, pbr], axis=1)
    ab = os.path.join(args.out, "ab_compare.png")
    Image.fromarray(strip).save(ab)
    print(f"DONE -> {ab}")


if __name__ == "__main__":
    main()
