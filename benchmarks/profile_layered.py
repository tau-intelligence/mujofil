"""Break down WHERE the layered full-loop time goes, to find the physics<->render
integration cost. Splits one step into:
  physics | bounce (geom_xpos.numpy + host loop) | native layered render |
  python composite+tonemap

Also compares tonemap ON (cam_id>=0, per-world quantile loop) vs the shared-camera
path (cam_id<0, no per-world loop) to isolate the Python auto-exposure cost.

Run:
  systemd-run --user -p LimitAS=infinity --quiet --wait --pipe -- \
    bash -c 'cd ~/mujofil-warp && source .venv/bin/activate && \
      PYTHONPATH=$PWD MUJOFIL_WARP_BACKEND=gl python benchmarks/profile_layered.py'
"""
import time
import mujoco
import mujoco_warp as mjw
import warp as wp
import torch

from mujofil_warp import WarpRenderer, RendererConfig

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


def profile(res, nworld, cam_id, iters=30, warmup=8):
    mjm = mujoco.MjModel.from_xml_string(SCENE)
    ngeom = mjm.ngeom
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    host = [mujoco.MjData(mjm) for _ in range(nworld)]
    cfg = RendererConfig(); cfg.width = cfg.height = res
    cfg.batch_size = nworld; cfg.layered = True
    r = WarpRenderer(cfg); r.load_model(mjm)
    if cam_id < 0:
        r.set_free_camera(0, -3.0, 1.2, 0, 0, 0.4)

    tp = tb = tn = tc = 0.0

    def once(acc):
        nonlocal tp, tb, tn, tc
        t0 = time.perf_counter()
        mjw.step(m, d); wp.synchronize()
        t1 = time.perf_counter()
        gx = d.geom_xpos.numpy()
        gm = d.geom_xmat.numpy().reshape(nworld, ngeom, 9)
        for i, h in enumerate(host):
            h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]
        t2 = time.perf_counter()
        # native layered render only (raw dlpack, no python composite)
        ptrs = [int(h._address) for h in host]
        cap = r._r.render_batch_layered_dlpack(int(mjm._address), ptrs, cam_id)
        _ = torch.from_dlpack(cap)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        # full python path (composite + tonemap)
        _ = r.render_batch_layered(mjm, host, cam_id=cam_id)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if acc:
            tp += t1 - t0; tb += t2 - t1; tn += t3 - t2; tc += t4 - t3

    for _ in range(warmup): once(False)
    for _ in range(iters): once(True)
    r.close()
    f = 1000.0 / iters
    tot = (tp + tb + tn + tc) * f
    tag = "egocentric/tonemap ON" if cam_id >= 0 else "shared cam / tonemap OFF"
    print(f"  res {res} N={nworld}  [{tag}]  total {tot:.2f} ms/step")
    print(f"     physics            {tp*f:7.2f} ms")
    print(f"     bounce             {tb*f:7.2f} ms")
    print(f"     native layered     {tn*f:7.2f} ms")
    print(f"     python composite   {tc*f:7.2f} ms  <- tonemap+blend in torch")


if __name__ == "__main__":
    for res in (128,):
        for n in (64, 256, 1024):
            profile(res, n, cam_id=0)    # tonemap ON (per-world quantile loop)
            profile(res, n, cam_id=-1)   # tonemap OFF
            print()
