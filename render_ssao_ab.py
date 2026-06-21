"""Visual A/B: screen-space effects ON vs OFF, same warehouse environment.

Renders the SAME camera + scene twice -- once with SSAO + shadows + SSR enabled,
once with all three off (IBL + direct lighting + PBR materials kept) -- and
composites them side by side. This is the exact tradeoff the parallel-batch path
would make: the layered single-pass render keeps geometry + IBL + direct lighting
but cannot fold the per-view screen-space passes into one draw.

Each config renders in its OWN subprocess (two Filament engines in one process
can crash on teardown). Run:  python render_ssao_ab.py
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
A = os.path.join(WH, "assets")

# A detailed scene: chrome (SSR + reflections), gold/copper metals (IBL), a matte
# crate and rough spheres (SSAO contact), all on the floor under the racks (cast
# shadows). Arranged close together so contact AO + inter-object shadows show.
SCENE = """
<mujoco>
  <visual><global offwidth="1600" offheight="1200"/></visual>
  <asset>
    <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.05"/>
    <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.18"/>
    <material name="copper" rgba="0.95 0.6 0.45 1"  metallic="1.0" roughness="0.30"/>
    <material name="red"    rgba="0.80 0.18 0.15 1" metallic="0.0" roughness="0.55"/>
    <material name="crate"  rgba="0.55 0.40 0.25 1" metallic="0.0" roughness="0.85"/>
    <material name="white"  rgba="0.85 0.85 0.88 1" metallic="0.0" roughness="0.40"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    <body pos="-0.85 0.10 0.34"><freejoint/><geom type="sphere" size="0.34" material="chrome"/></body>
    <body pos="0.02 -0.35 0.26"><freejoint/><geom type="box" size="0.26 0.26 0.26" material="crate"/></body>
    <body pos="0.05 0.55 0.22"><freejoint/><geom type="sphere" size="0.22" material="gold"/></body>
    <body pos="0.75 -0.10 0.20"><freejoint/><geom type="sphere" size="0.20" material="copper"/></body>
    <body pos="0.95 0.55 0.16"><freejoint/><geom type="capsule" size="0.10 0.18" material="red"/></body>
    <body pos="-0.30 -0.70 0.16"><freejoint/><geom type="box" size="0.16 0.16 0.16" material="white"/></body>
    <camera name="cam0" pos="2.4 -2.4 1.5" xyaxes="0.71 0.71 0 -0.34 0.34 0.88"/>
  </worldbody>
</mujoco>
"""


def _warehouse(rr):
    import json
    rr.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
                os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        rr.load_glb(os.path.join(A, g))
    rr.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        rr.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)


def worker(effects, out_path):
    import mujofil
    os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR",
                          os.path.join(HERE, "mujofil_warp", "materials"))
    import numpy as np, mujoco, torch  # noqa
    from mujofil_warp import WarpRenderer, RendererConfig
    on = effects == "on"
    cfg = RendererConfig()
    cfg.width, cfg.height = 1200, 900
    cfg.batch_size = 1
    cfg.enable_ssao = on
    cfg.enable_shadows = on
    cfg.enable_ssr = on
    cfg.enable_msaa = True          # keep AA in BOTH (isolate the 3 effects)
    cfg.enable_bloom = False
    cfg.exposure = 1.0
    r = WarpRenderer(cfg)
    m = mujoco.MjModel.from_xml_string(SCENE)
    d = mujoco.MjData(m)
    # DROP the objects from a height so they fall under gravity and genuinely
    # settle ON the floor (real contact -> real contact AO + cast shadows). Lift
    # each body ~0.5 m above its rest pose, then step until kinetic energy dies.
    for b in range(m.njnt):
        d.qpos[7 * b + 2] += 0.5
    mujoco.mj_forward(m, d)
    for _ in range(600):
        mujoco.mj_step(m, d)
    r.load_model(m)
    _warehouse(r)
    r.sync_transforms(m, d)
    # Confirm the colliders engaged: contact count + settled heights + residual
    # speed (near 0 = at rest on the floor).
    speed = float(np.linalg.norm(d.qvel))
    zs = [round(float(d.qpos[7 * b + 2]), 3) for b in range(m.njnt)]
    print(f"contacts(ncon)={d.ncon}  settled z={zs}  |qvel|={speed:.4f}", flush=True)
    # Close, low, looking slightly DOWN at the objects resting on the floor, so
    # the floor fills the lower frame: contact AO, cast shadows, and floor
    # reflections are all in view. Tight framing keeps the busy warehouse behind.
    r.set_free_camera(1.7, -1.7, 0.85, 0.0, 0.0, 0.12)
    img = r.render()[..., :3].cpu().numpy()   # (H,W,3) uint8 upright
    from PIL import Image
    Image.fromarray(img).save(out_path)
    print(f"wrote {out_path} effects={effects}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i + 1], sys.argv[i + 2])
        return
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    paths = {}
    for eff in ("on", "off"):
        out = os.path.join(HERE, "out", f"ssao_{eff}.png")
        env = dict(os.environ)
        env["MUJOFIL_WARP_BACKEND"] = "gl"
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
        env["VF_MUJOCO_MATERIALS_DIR"] = os.path.join(HERE, "mujofil_warp", "materials")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", eff, out],
                           capture_output=True, text=True, env=env, cwd=HERE)
        ok = os.path.exists(out)
        print(f"  effects={eff}: {'OK' if ok else 'FAILED'}")
        if not ok:
            print(r.stderr.strip().splitlines()[-5:] if r.stderr else "(no stderr)")
        else:
            paths[eff] = out

    if len(paths) == 2:
        from PIL import Image, ImageDraw, ImageFont
        a = Image.open(paths["on"]); b = Image.open(paths["off"])
        W, H = a.size
        pad = 12
        canvas = Image.new("RGB", (W * 2 + pad * 3, H + pad * 2 + 44), (24, 24, 28))
        canvas.paste(a, (pad, pad + 44))
        canvas.paste(b, (W + pad * 2, pad + 44))
        dr = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        except Exception:
            font = ImageFont.load_default()
        dr.text((pad + 8, 12), "WITH SSAO + Shadows + SSR", fill=(120, 230, 140), font=font)
        dr.text((W + pad * 2 + 8, 12), "WITHOUT (IBL + direct light + PBR only)", fill=(240, 180, 110), font=font)
        comp = os.path.join(HERE, "out", "ssao_compare.png")
        canvas.save(comp)
        print(f"\nwrote comparison -> {comp}")


if __name__ == "__main__":
    main()
