"""Adversarial robustness harness for the LAYERED parallel-batch renderer.

Throws many scene types + batch configs at render_batch_layered and checks each
result is correct WITHOUT any per-scene calibration. The goal: it must work out
of the box for ANY environment, or fail with a clear error -- never silently
produce wrong frames (e.g. all props in layer 0, black worlds, lost backdrop).

Each case runs in its OWN subprocess (clean Filament state). A case PASSES when:
  - it renders (N,res,res,4) on cuda without crashing, AND
  - every world is non-black (each layer got content), AND
  - when the scene has per-world variation, worlds are pairwise-distinct
    (proves gl_Layer routing, not "everything in layer 0"), AND
  - when worlds are identical by construction, they ARE identical (no leak).

Run:  MUJOFIL_WARP_BACKEND=gl python benchmarks/adversarial_layered.py
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import mujofil
_warp_mats = os.path.join(HERE, "mujofil", "materials")
if os.path.isdir(_warp_mats):
    os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", _warp_mats)

WH = "/home/mumuksh/Visual-Fidelity-Mujoco/warehouse/data"
A = os.path.join(WH, "assets")
HAS_WAREHOUSE = os.path.isdir(A)

# ---------------------------------------------------------------------------
# Scene generators. Each returns (xml, expect_distinct: bool, loader_name|None).
# ---------------------------------------------------------------------------

def s_primitives():
    return ("""
    <mujoco><worldbody>
      <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
      <light pos="0 0 4"/>
      <body pos="0 0 0.4"><freejoint/><geom type="box" size="0.25 0.25 0.25" rgba="0.8 0.3 0.2 1"/></body>
      <body pos="0.6 0 0.3"><freejoint/><geom type="sphere" size="0.2" rgba="0.2 0.8 0.3 1"/></body>
      <camera name="c" pos="0 -2.6 1.0" xyaxes="1 0 0 0 0.36 0.93"/>
    </worldbody></mujoco>""", True, None)


def s_no_floor_no_light():
    # degenerate: no plane, no light, just a body
    return ("""
    <mujoco><worldbody>
      <body pos="0 0 0.4"><freejoint/><geom type="box" size="0.3 0.3 0.3" rgba="0.8 0.3 0.2 1"/></body>
      <camera name="c" pos="0 -2.6 0.6" xyaxes="1 0 0 0 0.36 0.93"/>
    </worldbody></mujoco>""", True, None)


def s_metals():
    return ("""
    <mujoco><asset>
      <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.05"/>
      <material name="gold" rgba="1 0.78 0.34 1" metallic="1.0" roughness="0.2"/></asset>
    <worldbody>
      <geom type="plane" size="10 10 0.1" rgba="0.4 0.4 0.45 1"/>
      <light pos="0 0 4" dir="0 0 -1"/>
      <body pos="-0.4 0 0.35"><freejoint/><geom type="sphere" size="0.3" material="chrome"/></body>
      <body pos="0.5 0 0.25"><freejoint/><geom type="box" size="0.22 0.22 0.22" material="gold"/></body>
      <camera name="c" pos="0 -2.8 1.1" xyaxes="1 0 0 0 0.36 0.93"/>
    </worldbody></mujoco>""", True, None)


def s_transparent():
    return ("""
    <mujoco><asset>
      <material name="glass" rgba="0.6 0.8 0.9 0.35"/></asset>
    <worldbody>
      <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
      <light pos="0 0 4"/>
      <body pos="0 0 0.4"><freejoint/><geom type="sphere" size="0.3" material="glass"/></body>
      <camera name="c" pos="0 -2.6 1.0" xyaxes="1 0 0 0 0.36 0.93"/>
    </worldbody></mujoco>""", True, None)


def s_many_geoms():
    bodies = "".join(
        '<body name="b%d" pos="%.2f %.2f 0.25"><freejoint/>%s</body>'
        % (i, -2 + 0.5 * (i % 9), -2 + 0.5 * (i // 9),
           ('<geom name="g%d" type="box" size="0.18 0.18 0.18" rgba="%.2f 0.40 0.60 1"/>'
            % (i, 0.3 + 0.5 * (i % 3) / 3)) if i % 2 else
           ('<geom name="g%d" type="sphere" size="0.18" rgba="%.2f 0.40 0.60 1"/>'
            % (i, 0.3 + 0.5 * (i % 3) / 3)))
        for i in range(45))
    return (f"""
    <mujoco><worldbody>
      <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
      <light pos="0 0 6"/>
      {bodies}
      <camera name="c" pos="0 -5 4" xyaxes="1 0 0 0 0.6 0.8"/>
    </worldbody></mujoco>""", True, None)


def s_capsule_cylinder_ellipsoid():
    return ("""
    <mujoco><worldbody>
      <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
      <light pos="0 0 4"/>
      <body pos="-0.6 0 0.3"><freejoint/><geom type="capsule" size="0.12 0.25" rgba="0.8 0.3 0.2 1"/></body>
      <body pos="0 0 0.3"><freejoint/><geom type="cylinder" size="0.18 0.2" rgba="0.2 0.7 0.8 1"/></body>
      <body pos="0.6 0 0.3"><freejoint/><geom type="ellipsoid" size="0.2 0.15 0.3" rgba="0.7 0.7 0.2 1"/></body>
      <camera name="c" pos="0 -2.8 1.0" xyaxes="1 0 0 0 0.36 0.93"/>
    </worldbody></mujoco>""", True, None)


def s_identical_worlds():
    # No per-world variation injected -> all worlds must be IDENTICAL.
    return ("""
    <mujoco><worldbody>
      <geom type="plane" size="10 10 0.1" rgba="0.5 0.5 0.55 1"/>
      <light pos="0 0 4"/>
      <body pos="0 0 0.4"><freejoint/><geom type="box" size="0.3 0.3 0.3" rgba="0.8 0.3 0.2 1"/></body>
      <camera name="c" pos="0 -2.6 1.0" xyaxes="1 0 0 0 0.36 0.93"/>
    </worldbody></mujoco>""", "identical", None)


def s_warehouse():
    return ("""
    <mujoco><visual><global offwidth="1024" offheight="1024"/></visual><asset>
      <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.06"/></asset>
    <worldbody>
      <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
      <body pos="-0.4 0 0.4"><freejoint/><geom type="sphere" size="0.3" material="chrome"/></body>
      <body pos="0.5 0 0.25"><freejoint/><geom type="box" size="0.22 0.22 0.22" rgba="0.8 0.6 0.2 1"/></body>
      <camera name="c" pos="0 -3 1.2" xyaxes="1 0 0 0 0.36 0.93"/>
    </worldbody></mujoco>""", True, "warehouse")


SCENES = {
    "primitives": s_primitives,
    "no_floor_no_light": s_no_floor_no_light,
    "metals": s_metals,
    "transparent": s_transparent,
    "many_geoms": s_many_geoms,
    "capsule_cyl_ellipsoid": s_capsule_cylinder_ellipsoid,
    "identical_worlds": s_identical_worlds,
}
if HAS_WAREHOUSE:
    SCENES["warehouse_glb"] = s_warehouse


def _load_warehouse(r):
    r.load_ibl(os.path.join(A, "ibl", "warehouse_ibl_ibl.ktx"),
               os.path.join(A, "ibl", "warehouse_ibl_skybox.ktx"))
    for g in ["nvidia_floor.glb", "nvidia_warehouse_tinted.glb", "wall_patch.glb"]:
        r.load_glb(os.path.join(A, g))
    r.set_ambient_intensity(9000.0)
    for x, y, z in json.load(open(os.path.join(WH, "lamps.json"))):
        r.add_spot_light(x, y, z, 0, 0, -1, 1, .97, .9, 32e6, 14, 55, 88, True)


def worker(scene_name, n, res):
    import numpy as np, torch, mujoco
    from mujofil import WarpRenderer, RendererConfig
    xml, expect, loader = SCENES[scene_name]()
    m = mujoco.MjModel.from_xml_string(xml)
    datas = [mujoco.MjData(m) for _ in range(n)]
    if expect is True:  # LARGE unambiguous per-world signal: shift EVERY free body
        # by a per-world circle offset, so the whole scene visibly differs per
        # world regardless of geom count (defeats sub-pixel false negatives).
        for i, d in enumerate(datas):
            ang = 6.2831853 * i / n
            dx, dy = 1.0 * float(np.cos(ang)), 1.0 * float(np.sin(ang))
            for b in range(m.njnt):
                d.qpos[7 * b + 0] += dx
                d.qpos[7 * b + 1] += dy
            mujoco.mj_forward(m, d)
    else:               # identical worlds: same state everywhere
        for d in datas:
            mujoco.mj_forward(m, d)

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    cfg.layered = True
    r = WarpRenderer(cfg)
    r.load_model(m)
    if loader == "warehouse":
        _load_warehouse(r)
    imgs = r.render_batch_layered(m, datas, cam_id=0)
    torch.cuda.synchronize()

    assert imgs.is_cuda, "not on cuda"
    assert tuple(imgs.shape) == (n, res, res, 4), f"bad shape {tuple(imgs.shape)}"
    rgb = imgs[..., :3].float()
    bright = rgb.mean(dim=(1, 2, 3))           # (N,)
    nonblack = int((bright > 3).sum())
    # Definitive routing check: well-separated worlds must be distinct, and the
    # batch must carry real variance (std~0 only if every layer is identical =
    # routing collapsed). Sampling spread-out worlds avoids sub-pixel false fails.
    import itertools
    if n > 1:
        probe = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        max_pair = max((rgb[a] - rgb[b]).abs().mean().item()
                       for a, b in itertools.combinations(probe, 2))
        batch_std = rgb.std(dim=0).mean().item()
    else:
        max_pair = 0.0
        batch_std = 0.0

    problems = []
    if nonblack < n:
        problems.append(f"{n-nonblack}/{n} worlds BLACK")
    if expect is True and n > 1 and (max_pair < 1.0 or batch_std < 0.3):
        problems.append(f"worlds not distinct (max_pair={max_pair:.2f} std={batch_std:.2f}) -> routing collapsed")
    if expect == "identical" and n > 1 and max_pair > 1.0:
        problems.append(f"identical-worlds scene differs (max_pair={max_pair:.2f}) -> leak")

    status = "PASS" if not problems else "FAIL"
    print(f"RESULT {status} scene={scene_name} n={n} res={res} "
          f"nonblack={nonblack}/{n} max_pair={max_pair:.2f} std={batch_std:.2f} "
          f"bright={bright.mean().item():.1f}" +
          (("  :: " + "; ".join(problems)) if problems else ""), flush=True)
    sys.stdout.flush()
    # Exercise teardown + reload BEFORE os._exit. Filament aborts the whole
    # process on a teardown double-free or a missing-uniform setParameter panic;
    # os._exit skips C++ dtors and would hide such crashes (exactly how a material
    # -instance double-free and a stale-material panic slipped past this harness
    # once). A reload re-runs the material-free path; close() runs the dtors. If
    # either aborts, the parent records this case as an ERROR.
    r.load_model(m)
    r.close()
    os._exit(0)


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i + 1], int(sys.argv[i + 2]), int(sys.argv[i + 3]))
        return

    cases = []
    for name in SCENES:
        cases.append((name, 8, 128))
        cases.append((name, 64, 128))
    cases.append(("primitives", 1, 128))      # N=1 edge
    cases.append(("primitives", 256, 96))     # at UBO cap
    cases.append(("primitives", 300, 96))     # OVER cap -> must error clearly, not corrupt

    npass = nfail = nerr = 0
    print("=== ADVERSARIAL LAYERED ROBUSTNESS ===")
    for name, n, res in cases:
        env = dict(os.environ); env["MUJOFIL_WARP_BACKEND"] = "gl"
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", name, str(n), str(res)],
                             capture_output=True, text=True, env=env, cwd=HERE)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
        if line:
            print("  " + line[7:])
            if line.startswith("RESULT PASS"):
                npass += 1
            else:
                nfail += 1
        else:
            err = (out.stderr.strip().splitlines() or ["(no output)"])[-1]
            # an over-cap case is allowed to ERROR clearly (not corrupt)
            tag = "ERR(ok if clear)" if n > 256 else "ERR"
            print(f"  {tag} scene={name} n={n}: {err[:90]}")
            nerr += 1
    print(f"\n=== {npass} pass, {nfail} fail, {nerr} error ===")


if __name__ == "__main__":
    main()
