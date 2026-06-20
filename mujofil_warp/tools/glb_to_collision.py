"""Generalised, deterministic GLB -> collision-MJCF converter.

Photoreal environments enter MuJoFil as GLB *backdrops* (rendered by Filament,
invisible to physics). This tool produces the missing PHYSICS half: a collision-
only MJCF that is perfectly aligned with the rendered GLB, so a robot actually
stands on the floor, bumps into walls and can't fall through anything — while the
GLB still provides every pixel you see.

How the two halves stay invisible to each other and aligned:
  * The collision geoms are emitted at ``group="3"``. MuJoFil's SceneBridge skips
    every geom with group >= 3 (collision-only), so they never render — the GLB
    is the sole visual. MuJoCo's own viewer also hides group 3 by default.
  * Collision geometry is baked in the SAME world frame as the GLB by applying
    the identical column-major 4x4 visual transform used by ``load_glb_xform``.
    So "what you see" (GLB) and "what you hit" (MJCF) occupy the same space.

Collision strategy (deterministic, performance-tiered):
  floor   : the dominant up-facing horizontal surface under the scene becomes an
            infinite ``plane`` geom. Cheapest possible; guarantees nothing falls
            through and robots stand/walk. (always on)
  walls   : axis-aligned ``box`` geoms at the environment's XY bounds (+ ceiling)
            keep robots inside the room. A handful of boxes -> negligible cost.
            (--walls)
  meshes  : every render submesh -> its convex hull as an invisible ``mesh`` geom,
            so pillars/props/furniture are collidable. Heavier (convexification at
            load); concave shells become convex, so best for convex-ish props.
            (--meshes)

Output: <out>/<name>_collision.xml (+ <out>/<name>_meshes/ if --meshes). The XML
is a self-contained <mujoco> you ``<include>`` into any robot scene.

Usage:
  python -m scripts.glb_to_collision <scene_key> [--walls] [--meshes] [--ceiling]
  python -m scripts.glb_to_collision --glb path.glb --xform "16 floats" --name foo
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 8 sign combinations for the corners of a box (half-extent multipliers).
_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                  dtype=np.float64)


def _xform_matrix(col_major16):
    """Column-major 16 -> 4x4 row-major numpy (same convention as load_glb_xform)."""
    m = np.asarray(col_major16, dtype=np.float64).reshape(4, 4).T
    return m


def _scene_parts(scene):
    """Per-geometry parts of a trimesh Scene, baked into world space.

    Replaces the deprecated ``Scene.dump(concatenate=False)`` (removed in trimesh
    2025) with the documented graph-walk equivalent, which applies each node's
    world transform. Falls back to ``dump`` on older trimesh; a plain Trimesh is
    returned as a single part."""
    if not isinstance(scene, trimesh.Scene):
        return [scene]
    parts = []
    try:
        for node in scene.graph.nodes_geometry:
            T, gname = scene.graph[node]
            g = scene.geometry[gname].copy()
            g.apply_transform(T)
            parts.append(g)
        if parts:
            return parts
    except Exception:
        pass
    return scene.dump(concatenate=False)  # legacy fallback


def _load_world_mesh(glb_path, M):
    """Load a GLB as one mesh, baked into world space by the visual transform."""
    g = trimesh.load(glb_path, force="mesh", process=False)
    v = trimesh.transformations.transform_points(g.vertices, M)
    return v, g.faces


def _detect_floor_z(v, faces, center=(0.0, 0.0), base_z=0.0, footprint_r=1.2):
    """World-Z of the ground floor the robot stands on, robust to scene scale,
    position and layout WITHOUT any manual calibration. (``center``/``footprint_r``
    are accepted for call-site compatibility but unused -- detection is
    position-independent.) ``base_z`` is the robot mount plane (default 0): the
    scene's authoring/conversion places the walkable floor at this height, so it
    is the reference the floor is matched against.

    Method = area-weighted histogram of near-HORIZONTAL surfaces over the WHOLE
    scene, cluster them, then pick the significant deck NEAREST ``base_z``.
    Rationale:
      * Horizontal + area-weighted -> walls (vertical) and thin props/clutter
        contribute almost nothing, so the floor and other broad decks dominate.
      * Count a face as horizontal by |nz| (BOTH up- and down-facing), not just
        up-facing: many GLB floors ship with flipped or double-sided normals, so
        an up-only test silently MISSES the ground and latches onto an elevated
        deck/mezzanine instead (this is why the gallery/office floor was detected
        at z=3.1 instead of 0). A down-facing slab underside sits at the same
        height as its top, so including it never moves the floor.
      * "Significant deck NEAREST base_z" (not merely the LOWEST deck) -> the
        robot is mounted at base_z and stands ON the floor at that height. A pure
        "lowest" rule breaks on scenes with a raised walkable platform: e.g.
        Sponza's robot stands on the central dais at z~0 while the surrounding
        ground sinks to z=-0.95, so "lowest" wrongly returned -0.95. Nearest-to-
        base_z returns the dais. On single-level scenes the nearest deck IS the
        lowest, so this is a strict generalisation. Ties break toward the LOWER
        deck (you rest on top of the lower of two equally-close surfaces).
      * No raycast and no origin/footprint assumption -> independent of where the
        scene sits in XY (a single ray under the origin used to latch onto a
        catwalk when the origin happened to sit under one).
    A flat scene with no clear horizontal surface falls back to the lowest vertex.
    """
    tri = v[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1) + 1e-12
    nz = n[:, 2] / ln
    area = 0.5 * ln
    cz = tri.mean(axis=1)[:, 2]
    horiz = np.abs(nz) > 0.85  # near-horizontal, EITHER normal direction

    if horiz.sum() >= 8:
        h, w = cz[horiz], area[horiz]
        zlo, zhi = float(h.min()), float(h.max())
        # 5cm bins scale to any vertical extent; a flat deck lands in one bin and
        # adjacent bins of a slightly-uneven deck both clear the threshold.
        nb = max(12, int((zhi - zlo) / 0.05) + 1)
        hist, edges = np.histogram(h, bins=np.linspace(zlo, zhi + 1e-4, nb),
                                   weights=w)
        if hist.max() > 0:
            sig = np.nonzero(hist >= 0.25 * hist.max())[0]
            if sig.size:
                centers = (edges[sig] + edges[sig + 1]) / 2.0
                # floor = significant deck nearest the robot mount plane, ties
                # broken toward the lower deck (key sorts by distance, then z).
                best = min(centers, key=lambda z: (abs(z - base_z), z))
                return float(best)
    # last resort (terrain/sculpted scene with no broad horizontal deck).
    return float(v[:, 2].min())


def _world_bounds(v):
    return v.min(axis=0), v.max(axis=0)


def _obb_world(part, M):
    """Oriented (or axis-aligned, see below) bounding box of a GLB submesh in
    world space. Returns (pos[3], quat_wxyz[4], half_extents[3], extents[3],
    center[3]) or None.

    A box is the cheapest, most reliable MuJoCo collision primitive (no mesh edge
    cases). We PREFER a minimum-volume oriented box (it hugs rotated furniture
    far better), but that needs trimesh's convex-hull path which depends on
    scipy. If scipy is missing -- or the hull is degenerate/coplanar -- we fall
    back to a world-axis-aligned box instead of giving up. Returning None there
    would silently drop EVERY prop (a scene with no furniture collision), so the
    AABB fallback is what keeps prop collision working out of the box on any
    machine, with or without the optional scientific stack installed.
    """
    pv = trimesh.transformations.transform_points(part.vertices, M)
    pos = quat = ext = None
    try:
        mesh = trimesh.Trimesh(vertices=pv, faces=part.faces, process=False)
        obb = mesh.bounding_box_oriented
        T = np.asarray(obb.primitive.transform, dtype=np.float64)
        e = np.asarray(obb.primitive.extents, dtype=np.float64)
        if e is not None and not np.any(e <= 1e-4):
            pos = T[:3, 3]
            quat = trimesh.transformations.quaternion_from_matrix(T)  # (w,x,y,z)
            ext = e
    except Exception:
        pass  # scipy missing / coplanar / degenerate -> AABB fallback below
    if ext is None:
        # world-axis-aligned fallback: identity rotation, box = vertex bounds.
        lo = pv.min(axis=0)
        hi = pv.max(axis=0)
        ext = hi - lo
        if np.any(ext <= 1e-4):
            return None
        pos = (lo + hi) / 2.0
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w,x,y,z) identity
    return pos, quat, ext / 2.0, ext, pos


def build_mjcf(glb_path, M, name, out_dir, walls=True, props=True, ceiling=False,
               wall_thick=0.2, friction=(1.0, 0.005, 0.0001),
               prop_min_vol=0.01, prop_max_extent=6.0, max_props=80,
               robot_clear_xy=0.45, robot_clear_z=1.4, floor_z=None):
    v, faces = _load_world_mesh(glb_path, M)
    # ``floor_z`` lets the caller pin the ground plane height for scenes where the
    # auto-detector (raycast + lowest-significant-surface) can't be trusted
    # (e.g. multi-level/mezzanine GLBs). None = auto-detect.
    if floor_z is None:
        floor_z = _detect_floor_z(v, faces)
    else:
        floor_z = float(floor_z)
    bmin, bmax = _world_bounds(v)
    cx, cy = (bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2
    sx, sy = (bmax[0] - bmin[0]) / 2, (bmax[1] - bmin[1]) / 2
    top_z = float(bmax[2])
    fr = " ".join(str(x) for x in friction)

    os.makedirs(out_dir, exist_ok=True)
    geom_lines = []

    # Collision flags rationale (kept on EVERY env geom):
    #   group=3      -> invisible to MuJoFil (renderer skips group>=3) and MuJoCo viewer.
    #   condim=3     -> sliding friction only (no expensive torsional/rolling terms).
    #   contype/conaffinity at 1/1: all env geoms live in <worldbody> (body 0), so
    #   MuJoCo's same-body rule prunes EVERY env-vs-env pair for free; only the robot
    #   tests against the environment. Zero static-static cost, no bitmask needed.
    CFLAGS = 'group="3" contype="1" conaffinity="1" condim="3"'

    # --- floor plane (always) -------------------------------------------------
    geom_lines.append(
        f'    <geom name="{name}_floor" type="plane" pos="{cx:.4f} {cy:.4f} {floor_z:.4f}" '
        f'size="{max(sx, sy) + 5:.2f} {max(sx, sy) + 5:.2f} 0.1" '
        f'{CFLAGS} friction="{fr}" rgba="0 0 0 0"/>')

    # --- bounding walls (DEFAULT) --------------------------------------------
    if walls:
        h = max(top_z - floor_z, 2.0)
        zc = floor_z + h / 2
        t = wall_thick
        defs = [
            ("xlo", bmin[0] - t, cy, t, sy + t, h),
            ("xhi", bmax[0] + t, cy, t, sy + t, h),
            ("ylo", cx, bmin[1] - t, sx + t, t, h),
            ("yhi", cx, bmax[1] + t, sx + t, t, h),
        ]
        for tag, px, py, hx, hy, hz in defs:
            geom_lines.append(
                f'    <geom name="{name}_wall_{tag}" type="box" '
                f'pos="{px:.4f} {py:.4f} {zc:.4f}" size="{hx:.4f} {hy:.4f} {hz/2:.4f}" '
                f'{CFLAGS} rgba="0 0 0 0"/>')
        if ceiling:
            geom_lines.append(
                f'    <geom name="{name}_ceiling" type="box" '
                f'pos="{cx:.4f} {cy:.4f} {top_z + t:.4f}" size="{sx+t:.4f} {sy+t:.4f} {t:.4f}" '
                f'{CFLAGS} rgba="0 0 0 0"/>')

    # --- per-submesh oriented-bounding-BOX collision for PROPS (DEFAULT) ------
    # Every interior asset (tables, couches, shelves, crates...) becomes its own
    # oriented box so the robot can bump/rest on it. Boxes are the cheapest, most
    # reliable collision primitive. Floor/wall/ceiling sheets are skipped (already
    # covered); tiny clutter (< prop_min_vol) and the scene shell are skipped.
    n_props = 0
    if props:
        scene = trimesh.load(glb_path, process=False)
        parts = _scene_parts(scene)
        scene_vol = float((bmax[0]-bmin[0]) * (bmax[1]-bmin[1]) * max(top_z-floor_z, 0.1))
        for idx, part in enumerate(parts):
            if len(part.faces) < 4 or n_props >= max_props:
                continue
            ob = _obb_world(part, M)
            if ob is None:
                continue
            pos, quat, half, ext, center = ob
            vol = float(ext[0] * ext[1] * ext[2])
            # world-axis-aligned bounds of the OBB centre for shell/floor tests
            zc = float(center[2])
            if vol < prop_min_vol:                       # clutter
                continue
            if max(ext) > prop_max_extent or vol > 0.40 * scene_vol:   # scene shell
                continue
            # skip flat floor/ceiling sheets (thin + at floor or ceiling height)
            if min(ext) < 0.10 and (abs(zc - floor_z) < 0.18 or abs(zc - top_z) < 0.35):
                continue
            # skip thin perimeter wall panels (thin + near the room edge in X or Y)
            edge = (abs(center[0]-bmin[0]) < 0.5 or abs(center[0]-bmax[0]) < 0.5 or
                    abs(center[1]-bmin[1]) < 0.5 or abs(center[1]-bmax[1]) < 0.5)
            if min(ext) < 0.18 and edge:
                continue
            # skip props overlapping the ROBOT keep-out column at the world origin
            # (where the robot is mounted): a box the robot spawns inside causes
            # persistent penetration contacts every step = a big, useless physics
            # cost. The robot can still REACH props outside this small footprint.
            R = trimesh.transformations.quaternion_matrix(quat)[:3, :3]
            cs = center + (R @ (half[:, None] * _SIGNS.T)).T   # 8 world corners
            ax_min, ax_max = cs.min(axis=0), cs.max(axis=0)
            if (ax_min[0] < robot_clear_xy and ax_max[0] > -robot_clear_xy and
                    ax_min[1] < robot_clear_xy and ax_max[1] > -robot_clear_xy and
                    ax_min[2] < floor_z + robot_clear_z):
                continue
            q = " ".join(f"{x:.5f}" for x in quat)
            geom_lines.append(
                f'    <geom name="{name}_prop_{idx}" type="box" '
                f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" quat="{q}" '
                f'size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}" '
                f'{CFLAGS} rgba="0 0 0 0"/>')
            n_props += 1

    xml = (
        f'<mujoco model="{name}_collision">\n'
        f'  <!-- Collision-only proxy for the {name} GLB backdrop. group=3 -> invisible\n'
        f'       in MuJoFil (renderer skips group>=3) and in MuJoCo viewer. Aligned to\n'
        f'       the GLB by the same visual transform. Floor z = {floor_z:.4f} (world).\n'
        f'       floor + walls + {n_props} prop boxes. All in worldbody -> env-env\n'
        f'       contacts auto-pruned (same-body rule); only the robot collides. -->\n'
        f'  <worldbody>\n'
        f'{chr(10).join(geom_lines)}\n'
        f'  </worldbody>\n'
        f'</mujoco>\n'
    )
    out_path = os.path.join(out_dir, f"{name}_collision.xml")
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"floor_z={floor_z:.4f}  bounds X[{bmin[0]:.1f},{bmax[0]:.1f}] "
          f"Y[{bmin[1]:.1f},{bmax[1]:.1f}]  walls={walls} props={n_props} -> {out_path}")
    return out_path


def ensure_collision(glb_path, M, name, out_dir=None, **kw):
    """Bake-once cache. Returns the collision XML path, regenerating ONLY if the
    GLB or transform changed (content hash). Call this when an environment is
    REGISTERED (offline/background) so the training pipeline merely <include>s the
    cached XML -- zero collision-baking cost on the hot path.
    """
    import hashlib
    out_dir = out_dir or os.path.join(os.getcwd(), "collision")
    os.makedirs(out_dir, exist_ok=True)
    xml_path = os.path.join(out_dir, f"{name}_collision.xml")
    stamp_path = os.path.join(out_dir, f"{name}_collision.stamp")
    key = hashlib.md5()
    key.update(str(os.path.getmtime(glb_path)).encode())
    key.update(np.asarray(M, dtype=np.float64).tobytes())
    key.update(repr(sorted(kw.items())).encode())
    digest = key.hexdigest()
    if os.path.exists(xml_path) and os.path.exists(stamp_path):
        if open(stamp_path).read().strip() == digest:
            return xml_path  # cache hit -> no baking
    build_mjcf(glb_path, M, name, out_dir, **kw)
    with open(stamp_path, "w") as f:
        f.write(digest)
    return xml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", help="scene key from trailer.scenes")
    ap.add_argument("--glb")
    ap.add_argument("--xform", help="16 space-separated floats (column-major)")
    ap.add_argument("--name")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "collision"))
    ap.add_argument("--no-walls", dest="walls", action="store_false")
    ap.add_argument("--no-props", dest="props", action="store_false")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="pin the ground-plane Z (skip auto-detect; useful for "
                         "multi-level scenes where detection picks a mezzanine)")
    ap.set_defaults(walls=True, props=True)
    args = ap.parse_args()

    if args.scene:
        sys.path.insert(0, ROOT)
        try:
            from trailer.scenes import SCENES
        except ImportError:
            ap.error("the scene-key shortcut needs the project's "
                     "trailer.scenes (dev checkout); use --glb/--xform "
                     "instead when running the packaged tool")
            return
        s = SCENES[args.scene]
        glb = s["glb"]
        M = _xform_matrix(s["xform"])
        name = args.name or args.scene
    else:
        if not (args.glb and args.xform):
            ap.error("provide a scene key, or --glb and --xform")
        glb = args.glb
        M = _xform_matrix([float(x) for x in args.xform.split()])
        name = args.name or os.path.splitext(os.path.basename(glb))[0]

    build_mjcf(glb, M, name, args.out, walls=args.walls, props=args.props,
               ceiling=args.ceiling, floor_z=args.floor_z)


if __name__ == "__main__":
    main()
