"""Battle-test the LAYERED parallel-batch renderer on EVERY real environment we
have downloaded -- mesh-heavy MJCF robot/warehouse scenes AND the large photoreal
GLB backdrops (sponza, cafe, gallery, neon street, sci-fi, izakaya, chess, living
room, NVIDIA + Isaac warehouses). The point is to prove the renderer works out of
the box on ANY environment of any size/scale, with NO per-scene calibration, and
tears down cleanly (no double-free / no missing-uniform panic).

Each case runs in its OWN subprocess (clean Filament state + crash isolation: a
panic in one scene can't take down the rest). A case PASSES when:
  * it loads + renders (N,res,res,4) on cuda without crashing, AND
  * every world is non-black (each array layer received content), AND
  * the per-world variation makes worlds pairwise-distinct (gl_Layer routing
    really fanned the batch out -- not "everything in layer 0"), AND
  * the renderer closes + reloads WITHOUT aborting (teardown is exercised here;
    os._exit would otherwise hide dtor double-frees / material panics).

Run:  MUJOFIL_WARP_BACKEND=gl python benchmarks/battle_test_envs.py
"""
import os
import sys
import itertools
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
_warp_mats = os.path.join(HERE, "mujofil", "materials")
if os.path.isdir(_warp_mats):
    os.environ.setdefault("VF_MUJOCO_MATERIALS_DIR", _warp_mats)

VFM = "/home/mumuksh/Visual-Fidelity-Mujoco"
SCN = os.path.join(VFM, "assets", "scenes")
ENV = os.path.join(VFM, "assets", "envmaps")
STUDIO = (os.path.join(ENV, "studio_ibl", "studio_ibl_ibl.ktx"),
          os.path.join(ENV, "studio_ibl", "studio_ibl_skybox.ktx"))
WH_IBL = (os.path.join(VFM, "warehouse", "data", "assets", "ibl", "warehouse_ibl_ibl.ktx"),
          os.path.join(VFM, "warehouse", "data", "assets", "ibl", "warehouse_ibl_skybox.ktx"))

# (key, kind, target, ibl_or_None). kind="mjcf" -> load a complete model file;
# kind="glb" -> load that GLB as a static backdrop + IBL and composite per-world
# objects over it. ibl=None -> studio (a neutral set that always exists).
CASES = [
    ("franka",        "mjcf", os.path.join(VFM, "assets/models/franka_fr3_table.xml"), None),
    ("warehouse_mjcf","mjcf", os.path.join(VFM, "warehouse/mjcf/warehouse.xml"),       None),
    ("sponza",        "glb",  os.path.join(SCN, "sponza.glb"),       STUDIO),
    ("living",        "glb",  os.path.join(SCN, "living.glb"),       STUDIO),
    ("cafe",          "glb",  os.path.join(SCN, "cafe.glb"),         STUDIO),
    ("gallery",       "glb",  os.path.join(SCN, "gallery.glb"),      STUDIO),
    ("neon_street",   "glb",  os.path.join(SCN, "neon_street.glb"),  STUDIO),
    ("scifi_dark",    "glb",  os.path.join(SCN, "scifi_dark.glb"),   STUDIO),
    ("scifi_club",    "glb",  os.path.join(SCN, "scifi_club.glb"),   STUDIO),
    ("izakaya",       "glb",  os.path.join(SCN, "izakaya.glb"),      STUDIO),
    ("chess",         "glb",  os.path.join(SCN, "chess.glb"),        STUDIO),
    ("nvidia_wh",     "glb",  os.path.join(VFM, "warehouse/data/assets/nvidia_warehouse_tinted.glb"), WH_IBL),
    ("isaac_wh",      "glb",  os.path.join(VFM, "output/usd_out/isaac_wh.glb"), WH_IBL),
]

# Per-world objects composited over a GLB backdrop: a few bright PBR primitives +
# a light + a camera. No floor plane (the GLB supplies its own ground). Objects
# sit on freejoints so the harness can shift the whole set per world.
OBJ_XML = """
<mujoco><asset>
  <material name="chrome" rgba="0.95 0.96 0.98 1" metallic="1.0" roughness="0.08"/>
  <material name="gold" rgba="1 0.78 0.34 1" metallic="1.0" roughness="0.25"/></asset>
<worldbody>
  <light pos="0 0 4" dir="0 0 -1"/>
  <body pos="-0.4 0 0.6"><freejoint/><geom type="sphere" size="0.3" material="chrome"/></body>
  <body pos="0.5 0 0.5"><freejoint/><geom type="box" size="0.25 0.25 0.25" material="gold"/></body>
  <body pos="0 0.5 0.45"><freejoint/><geom type="capsule" size="0.12 0.25" rgba="0.85 0.2 0.2 1"/></body>
  <camera name="c" pos="0 -3.2 1.3" xyaxes="1 0 0 0 0.36 0.93"/>
</worldbody></mujoco>"""


def _vary(m, datas, mujoco, np):
    """Shift every free body by a per-world circle offset so the whole scene
    differs per world regardless of geom count (image-scale signal). Returns the
    number of perturbable joints -- 0 means a purely static scene, for which
    identical worlds is the CORRECT result (nothing to fan out)."""
    n = len(datas)
    perturbable = 0
    for i, d in enumerate(datas):
        ang = 6.2831853 * i / n
        dx, dy = 0.9 * float(np.cos(ang)), 0.9 * float(np.sin(ang))
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                adr = m.jnt_qposadr[j]
                d.qpos[adr + 0] += dx
                d.qpos[adr + 1] += dy
                if i == 0:
                    perturbable += 1
            elif m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                   mujoco.mjtJoint.mjJNT_SLIDE):
                adr = m.jnt_qposadr[j]
                d.qpos[adr] += 0.35 * float(np.sin(ang + j))
                if i == 0:
                    perturbable += 1
        mujoco.mj_forward(m, d)
    return perturbable


def worker(key, kind, target, ibl_ibl, ibl_sky, n, res):
    import numpy as np, torch, mujoco
    from mujofil import WarpRenderer, RendererConfig

    if kind == "mjcf":
        m = mujoco.MjModel.from_xml_path(target)
    else:
        m = mujoco.MjModel.from_xml_string(OBJ_XML)
    datas = [mujoco.MjData(m) for _ in range(n)]
    perturbable = _vary(m, datas, mujoco, np)

    cfg = RendererConfig()
    cfg.width = cfg.height = res
    cfg.batch_size = n
    cfg.layered = True
    r = WarpRenderer(cfg)
    r.load_model(m)
    if kind == "glb":
        r.load_ibl(ibl_ibl, ibl_sky)
        r.load_glb(target)
        r.set_ambient_intensity(30000.0)

    # Use a fixed MJCF camera when one exists; otherwise frame the scene with a
    # free camera derived from the geom AABB (works for any scene scale, no
    # per-scene calibration -- e.g. warehouse.xml ships no <camera>).
    if m.ncam > 0:
        cam = 0
    else:
        cam = -1
        lo = datas[0].geom_xpos.min(axis=0) if m.ngeom else np.array([-1, -1, 0.])
        hi = datas[0].geom_xpos.max(axis=0) if m.ngeom else np.array([1, 1, 1.])
        c = (lo + hi) / 2.0
        rad = float(np.linalg.norm(hi - lo)) * 0.6 + 1.0
        r.set_free_camera(float(c[0]), float(c[1] - rad), float(c[2] + rad * 0.5),
                          float(c[0]), float(c[1]), float(c[2]))
    imgs = r.render_batch_layered(m, datas, cam_id=cam)
    torch.cuda.synchronize()

    assert imgs.is_cuda, "not on cuda"
    assert tuple(imgs.shape) == (n, res, res, 4), f"bad shape {tuple(imgs.shape)}"
    rgb = imgs[..., :3].float()
    bright = rgb.mean(dim=(1, 2, 3))
    nonblack = int((bright > 3).sum())
    probe = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    max_pair = max((rgb[a] - rgb[b]).abs().mean().item()
                   for a, b in itertools.combinations(probe, 2)) if n > 1 else 99
    batch_std = rgb.std(dim=0).mean().item() if n > 1 else 99

    problems = []
    if nonblack < n:
        problems.append(f"{n-nonblack}/{n} worlds BLACK")
    if perturbable > 0:
        # dynamic scene: worlds must fan out (routing really happened)
        if n > 1 and (max_pair < 1.0 or batch_std < 0.3):
            problems.append(f"worlds not distinct (max_pair={max_pair:.2f} std={batch_std:.2f})")
    else:
        # purely static scene (no movable joints): identical worlds is CORRECT
        if n > 1 and max_pair > 1.0:
            problems.append(f"static scene leaks (max_pair={max_pair:.2f})")

    # Exercise teardown + reload (aborts the process on a double-free / uniform
    # panic; os._exit below would hide it).
    r.load_model(m)
    r.close()

    status = "PASS" if not problems else "FAIL"
    print(f"RESULT {status} env={key} kind={kind} n={n} res={res} "
          f"nonblack={nonblack}/{n} max_pair={max_pair:.2f} std={batch_std:.2f} "
          f"bright={bright.mean().item():.1f} jnts={perturbable}" +
          (("  :: " + "; ".join(problems)) if problems else ""), flush=True)
    sys.stdout.flush()
    os._exit(0)


def main():
    if "--worker" in sys.argv:
        a = sys.argv
        i = a.index("--worker")
        worker(a[i+1], a[i+2], a[i+3], a[i+4] or None, a[i+5] or None,
               int(a[i+6]), int(a[i+7]))
        return

    print("=== BATTLE TEST: LAYERED ON ALL REAL ENVIRONMENTS ===", flush=True)
    npass = nfail = nerr = 0
    for key, kind, target, ibl in CASES:
        if not os.path.isfile(target):
            print(f"  SKIP env={key} (missing {target})", flush=True)
            continue
        ibl_ibl, ibl_sky = (ibl if ibl else ("", ""))
        for n in (8, 64):
            res = 96
            cmd = [sys.executable, os.path.abspath(__file__), "--worker",
                   key, kind, target, ibl_ibl, ibl_sky, str(n), str(res)]
            env = dict(os.environ, MUJOFIL_WARP_BACKEND="gl")
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines()
                         if l.startswith("RESULT")), None)
            if line is None:
                nerr += 1
                tail = (p.stderr.strip().splitlines() or ["<no output>"])[-1]
                print(f"  ERROR env={key} n={n} :: {tail}", flush=True)
            else:
                msg = line[len("RESULT "):]
                print("  " + msg, flush=True)
                if msg.startswith("PASS"):
                    npass += 1
                else:
                    nfail += 1
    print(f"=== {npass} pass, {nfail} fail, {nerr} error ===", flush=True)
    sys.exit(1 if (nfail or nerr) else 0)


if __name__ == "__main__":
    main()
