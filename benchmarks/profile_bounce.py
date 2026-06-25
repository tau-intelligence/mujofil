"""Profile where the per-step time actually goes in the ParallelScene path, to
decide whether the GPU->CPU transform 'bounce' is worth eliminating.

Breaks one env-step into: physics (mjw.step) | bounce (geom_xpos.numpy + the
per-world host-MjData write loop) | render (batched PBR -> torch.cuda).

Run:
  systemd-run --user -p LimitAS=infinity --quiet --wait --pipe -- \
    bash -c 'cd ~/mujofil-warp && source .venv/bin/activate && \
      PYTHONPATH=$PWD MUJOFIL_WARP_BACKEND=gl python benchmarks/profile_bounce.py'
"""
import os
import sys
import time

import mujoco
import mujoco_warp as mjw
import warp as wp
import torch

from mujofil import WarpRenderer

SCENE = """
<mujoco>
  <option timestep="0.004"/>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
    <material name="blue"   rgba="0.20 0.45 0.85 1" metallic="0.0" roughness="0.30"/>
    <material name="floor"  rgba="0.45 0.46 0.50 1" metallic="0.0" roughness="0.6"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"/>
    <body pos="-0.6 0 0.8"><freejoint/><geom type="sphere" size="0.24" material="chrome"/></body>
    <body pos="0.1 0 1.0"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
    <body pos="0.8 0.3 0.7"><freejoint/><geom type="sphere" size="0.22" material="blue"/></body>
    <camera name="cam0" pos="0 -3.0 1.05" xyaxes="1 0 0 0 0.33 0.94"/>
  </worldbody>
</mujoco>
"""


def profile(res, nworld, iters=40, warmup=8):
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    ngeom = mjm.ngeom
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    host = [mujoco.MjData(mjm) for _ in range(nworld)]
    r = WarpRenderer(width=res, height=res, batch_size=nworld, preset="train")
    r.load_model(mjm)

    t_phys = t_bounce = t_render = 0.0

    def once(acc):
        nonlocal t_phys, t_bounce, t_render
        t0 = time.perf_counter()
        mjw.step(m, d)
        wp.synchronize()
        t1 = time.perf_counter()
        gx = d.geom_xpos.numpy()
        gm = d.geom_xmat.numpy().reshape(nworld, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]
            h.geom_xmat[:] = gm[i]
        t2 = time.perf_counter()
        _ = r.render_batch(mjm, host, cam_id=0)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if acc:
            t_phys += t1 - t0
            t_bounce += t2 - t1
            t_render += t3 - t2

    for _ in range(warmup):
        once(False)
    for _ in range(iters):
        once(True)
    r.close()

    tot = t_phys + t_bounce + t_render
    f = 1000.0 / iters
    print(f"  res {res} N={nworld}:  total {tot*f:.2f} ms/step")
    print(f"     physics  {t_phys*f:6.2f} ms  ({100*t_phys/tot:4.1f}%)")
    print(f"     bounce   {t_bounce*f:6.2f} ms  ({100*t_bounce/tot:4.1f}%)   <- GPU->CPU transforms + host loop")
    print(f"     render   {t_render*f:6.2f} ms  ({100*t_render/tot:4.1f}%)")


if __name__ == "__main__":
    for res in (128, 256):
        print(f"\n### res {res} ###")
        for n in (64, 256, 1024):
            profile(res, n)
