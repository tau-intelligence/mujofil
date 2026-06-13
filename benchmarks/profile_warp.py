"""Profile the warp warehouse render path: render vs flush vs copy."""
import os, json, time, sys
import mujoco, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mujofil_warp import WarpRenderer, RendererConfig

WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
A = os.path.join(WH, "assets")
SCENE = """<mujoco><visual><global offwidth="1024" offheight="1024"/></visual>
 <asset><material name="c" rgba="0.9 0.9 0.95 1" metallic="1.0" roughness="0.1"/></asset>
 <worldbody><geom type="plane" size="20 20 0.1" group="3"/>
 <body pos="0 0 0.3"><freejoint/><geom type="sphere" size="0.24" material="c"/></body>
 <camera name="cam0" pos="0 -3 1.05" xyaxes="1 0 0 0 0.33 0.94"/></worldbody></mujoco>"""

m = mujoco.MjModel.from_xml_string(SCENE)
for res, N in [(128, 32), (256, 32), (512, 8)]:
    datas = [mujoco.MjData(m) for _ in range(N)]
    for d in datas:
        mujoco.mj_forward(m, d)
    cfg = RendererConfig(); cfg.width = cfg.height = res; cfg.batch_size = N
    cfg.enable_ssao = True; cfg.enable_shadows = True; cfg.enable_msaa = True; cfg.exposure = 1.6
    r = WarpRenderer(cfg); r.load_model(m)
    r.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
               os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        r.load_glb(os.path.join(A, g))
    r.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        r.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)
    for _ in range(5):
        r.render_batch(m, datas, 0); torch.cuda.synchronize()
    r.reset_profile(); ITER = 20
    t0 = time.perf_counter()
    for _ in range(ITER):
        r.render_batch(m, datas, 0); torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    p = r.profile(); fr = p["frames"]
    print(f"PROFILE {res}x{res} N={N}: {N*ITER/wall:.0f} cam/s | "
          f"per-batch render={p['render_ms']/ITER:.2f}ms flush={p['flush_ms']/ITER:.2f}ms "
          f"copy={p['copy_ms']/ITER:.2f}ms | frames/batch={fr/ITER:.0f} | wall/batch={wall/ITER*1e3:.2f}ms")
    del r
