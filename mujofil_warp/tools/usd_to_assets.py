"""USD -> (visual GLB + physics MJCF) dual converter.

Why USD is a *better* source than a bare GLB
--------------------------------------------
A GLB is a pure VISUAL container - it has no physics schema, so converting
``.usd -> .glb`` throws away exactly the data we care about (authored colliders,
rigid bodies, mass). USD, by contrast, carries BOTH halves natively:

  * render geometry + ``UsdPreviewSurface`` materials  -> the pixels
  * ``UsdPhysics`` schemas (CollisionAPI / MeshCollisionAPI / RigidBodyAPI /
    MassAPI / native Cube|Sphere|Capsule|Cylinder colliders)  -> the physics

So instead of *approximating* collision from a mesh (the oriented-box route in
``glb_to_collision.py``), we read the collision the author already specified and
emit it faithfully into MuJoCo. A couch authored in USD as seat-box + back-box +
two armrest-boxes comes through as that exact multi-box decomposition - no
concavity loss.

No extra dependencies, no quality loss
--------------------------------------
``UsdPreviewSurface`` was deliberately designed to mirror glTF's metallic-
roughness PBR model (same channels: baseColor / metallic / roughness / emissive /
normal / occlusion / opacity). So a faithful UsdPreviewSurface -> glTF mapping is
essentially lossless; ``trimesh`` is only the GLB *container writer*. The one case
that genuinely loses fidelity is an asset authored PURELY in a proprietary shader
(Omniverse MDL, RenderMan PxrSurface) with NO UsdPreviewSurface fallback - that is
rare (Omniverse/Sketchfab/PolyHaven exports include the preview surface) and is
reported per-material so you always know when (and only when) it happens.

Outputs (both normalised to MuJoCo's Z-up, metres world frame):
  <out>/<name>.glb                 visual, for MuJoFil / Filament
  <out>/<name>_collision.xml       physics, <include> into a robot scene
  <out>/<name>_meshes/*.obj        convex collision meshes (only if needed)

Physics emission
  * static colliders  -> group=3 geoms in <worldbody> (invisible, env-env
    contacts auto-pruned by the same-body rule - same contract as glb_to_collision)
  * RigidBodyAPI prims -> movable <body> with a <freejoint> the robot can push,
    mass taken from MassAPI when present (else MuJoCo computes it from geometry)
  * no authored collision at all -> fall back to the OBB baker on the GLB we just
    wrote, so you always get *something* collidable.

Usage:
  python -m scripts.usd_to_assets in.usd --name kitchen
  python -m scripts.usd_to_assets in.usdz --out assets/collision --no-movable
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys

import numpy as np
import trimesh

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf
except Exception as exc:  # pragma: no cover - clear message if pxr missing
    raise SystemExit(
        "OpenUSD Python bindings (pxr) are required for USD import.\n"
        "Install with:  pip install usd-core\n"
        f"(import error: {exc})")


@contextlib.contextmanager
def _quiet_usd():
    """Silence pxr's noisy C++-level stderr (e.g. dozens of 'Found material
    bindings ... but MaterialBindingAPI is not applied' warnings) at the file-
    descriptor level, since they bypass Python's logging. Set USD_TO_ASSETS_VERBOSE=1
    to keep them. Real Python exceptions still propagate."""
    if os.environ.get("USD_TO_ASSETS_VERBOSE"):
        yield
        return
    try:
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            os.dup2(saved, 2)
            os.close(saved)
        except Exception:
            pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from .glb_to_collision import build_mjcf as _obb_build_mjcf  # packaged
except ImportError:  # standalone / dev checkout
    from glb_to_collision import build_mjcf as _obb_build_mjcf

_EPS = 1e-9

# Default cap (px) for the longest side of any (UDIM) texture tile baked into the
# GLB. Omniverse source textures are often 4K x N-tile UDIM sets, so an un-capped
# warehouse stage produces a multi-hundred-MB GLB. 1024 keeps assets practical;
# raise it (e.g. 2048) for hero fidelity or lower it (512) for tiny assets via the
# ``--max-tile`` CLI flag / ``convert(max_tile=...)``. ``convert`` overwrites this.
_TEXTURE_MAX_TILE = 1024


# --------------------------------------------------------------------------- #
# math helpers                                                                #
# --------------------------------------------------------------------------- #
def _gf_to_np(m) -> np.ndarray:
    """Gf.Matrix4d (USD row-vector convention) -> 4x4 numpy in COLUMN-vector
    convention (so ``N @ p`` transforms a point), i.e. the transpose."""
    a = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    return a.T


def _decompose(N: np.ndarray):
    """4x4 affine -> (translation[3], quat_wxyz[4], scale[3]). Assumes no shear."""
    t = N[:3, 3].copy()
    L = N[:3, :3]
    scale = np.linalg.norm(L, axis=0)
    scale[scale < _EPS] = _EPS
    R = L / scale  # remove scale from each column
    if np.linalg.det(R) < 0:  # mirrored: flip one axis so it stays a rotation
        scale[0] = -scale[0]
        R = L / scale
    H = np.eye(4)
    H[:3, :3] = R
    quat = trimesh.transformations.quaternion_from_matrix(H)  # (w, x, y, z)
    return t, quat, np.abs(scale)


def _up_axis_matrix(stage) -> np.ndarray:
    """Global matrix taking the stage's units + up-axis to MuJoCo Z-up metres."""
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    S = np.eye(4) * mpu
    S[3, 3] = 1.0
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        # Y-up -> Z-up: rotate +90 deg about X  (x,y,z) -> (x,-z,y)
        Rx = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
        return Rx @ S
    return S


def _axis_quat(axis_token: str) -> np.ndarray:
    """USD gprim axis token -> quat rotating local +Z onto that axis (wxyz)."""
    if axis_token == "X":
        return trimesh.transformations.quaternion_from_matrix(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    if axis_token == "Y":
        return trimesh.transformations.quaternion_from_matrix(
            trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return np.array([1.0, 0.0, 0.0, 0.0])


def _qmul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


# --------------------------------------------------------------------------- #
# geometry extraction                                                         #
# --------------------------------------------------------------------------- #
def _triangulate(counts, indices):
    """Polygon faceVertexCounts/Indices -> (tri_index_rows, corner_order).

    Returns the triangle list as indices INTO the flattened corner stream so we
    can pull per-corner attributes (faceVarying uv / normals) consistently."""
    tris = []
    corner = 0
    for c in counts:
        base = corner
        for k in range(1, c - 1):
            tris.append((base, base + k, base + k + 1))
        corner += c
    return np.asarray(tris, dtype=np.int64)


def _mesh_to_trimesh(prim, N, with_attrs=True):
    """UsdGeom.Mesh -> trimesh.Trimesh in world (Z-up, metres) space.

    Expanded to non-indexed corners so faceVarying UVs / normals stay correct."""
    m = UsdGeom.Mesh(prim)
    pts = m.GetPointsAttr().Get()
    if not pts:
        return None
    counts = m.GetFaceVertexCountsAttr().Get()
    idx = m.GetFaceVertexIndicesAttr().Get()
    if not counts or not idx:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)

    local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M = N @ _gf_to_np(local)

    tris_in_corner = _triangulate(counts, idx)  # rows index the corner stream
    if len(tris_in_corner) == 0:
        return None
    corner_vert = idx                                  # vertex id per corner
    corner_pos = pts[corner_vert]                      # (Ncorner, 3)
    corner_pos_w = trimesh.transformations.transform_points(corner_pos, M)

    faces = tris_in_corner                             # already corner-indexed
    verts = corner_pos_w
    uv = normals = None
    if with_attrs:
        uv = _read_uv(prim, corner_vert)
        normals = _read_normals(prim, corner_vert, M)

    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if normals is not None and len(normals) == len(verts):
        tm.vertex_normals = normals
    return tm, uv


def _read_uv(prim, corner_vert):
    pv = UsdGeom.PrimvarsAPI(prim)
    for nm in ("st", "st0", "UVMap", "map1"):
        p = pv.GetPrimvar(nm)
        if p and p.HasValue():
            vals = np.asarray(p.Get(), dtype=np.float64)
            interp = p.GetInterpolation()
            ind = p.GetIndices() if p.IsIndexed() else None
            if interp == UsdGeom.Tokens.faceVarying:
                arr = vals[ind] if ind is not None else vals
                return arr[: len(corner_vert)]
            if interp == UsdGeom.Tokens.vertex or interp == UsdGeom.Tokens.varying:
                src = vals[ind] if ind is not None else vals
                return src[corner_vert]
            if interp == UsdGeom.Tokens.constant and len(vals):
                return np.tile(vals[0], (len(corner_vert), 1))
    return None


def _read_normals(prim, corner_vert, M):
    m = UsdGeom.Mesh(prim)
    na = m.GetNormalsAttr()
    if na and na.HasValue():
        vals = np.asarray(na.Get(), dtype=np.float64)
        interp = m.GetNormalsInterpolation()
        R = M[:3, :3]
        Rn = np.linalg.inv(R).T  # normal matrix
        if interp == UsdGeom.Tokens.faceVarying:
            n = vals[: len(corner_vert)]
        elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            n = vals[corner_vert]
        else:
            return None
        n = (Rn @ n.T).T
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln < _EPS] = 1.0
        return n / ln
    return None


# --------------------------------------------------------------------------- #
# material extraction (UsdPreviewSurface -> glTF PBR)                          #
# --------------------------------------------------------------------------- #
def _connected_texture_file(inp):
    """If a shader input is wired to a UsdUVTexture, return its resolved file."""
    if not inp or not inp.HasConnectedSource():
        return None
    src = inp.GetConnectedSource()
    if not src:
        return None
    shader = UsdShade.Shader(src[0].GetPrim())
    if not shader:
        return None
    fin = shader.GetInput("file")
    if not fin or fin.Get() is None:
        return None
    asset = fin.Get()
    path = asset.resolvedPath or asset.path
    return path or None


def _const_or_none(inp):
    if inp and not inp.HasConnectedSource() and inp.Get() is not None:
        return inp.Get()
    return None


def _load_img(path, max_side=None):
    if max_side is None:
        max_side = _TEXTURE_MAX_TILE
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image
        im = Image.open(path).convert("RGBA")
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        return im
    except Exception:
        return None


def _texture_template(inp):
    """An asset-valued shader input -> absolute path (may still contain a <UDIM>
    token). Resolves relative authored paths against the layer they live in, so it
    works through references (the texture is anchored to the asset, not the stage).
    """
    if not inp or inp.Get() is None:
        return None
    a = inp.Get()
    if getattr(a, "resolvedPath", None):
        return a.resolvedPath
    authored = a.path
    if not authored:
        return None
    if os.path.isabs(authored):
        return authored
    try:
        for spec in inp.GetAttr().GetPropertyStack(Usd.TimeCode.Default()):
            if spec.layer and spec.layer.realPath:
                base = os.path.dirname(spec.layer.realPath)
                return os.path.normpath(os.path.join(base, authored))
    except Exception:
        pass
    return authored


def _load_udim_atlas(template, linear=False, max_tile=None):
    """Load a (possibly UDIM) texture as ONE atlas image + its (nu, nv) tiling.

    glTF has no UDIM concept, so a UDIM set (tiles 1001, 1002, ... where
    tile = 1001 + tu + 10*tv) is packed into a single nu x nv atlas and the mesh
    UVs are later divided by (nu, nv) to match. A non-UDIM path returns ((image),
    (1, 1)). Tiles are downsized to ``max_tile`` px so a 4-tile 4k set doesn't
    explode the GLB."""
    if max_tile is None:
        max_tile = _TEXTURE_MAX_TILE
    if template is None:
        return None, (1, 1)
    if "<UDIM>" not in template and "<udim>" not in template:
        img = _load_img(template, max_side=max_tile)
        return img, (1, 1)
    import glob
    import re
    tok = "<UDIM>" if "<UDIM>" in template else "<udim>"
    pat = template.replace(tok, "*")
    rx = re.compile(re.escape(template).replace(re.escape(tok), r"(\d{4})"))
    tiles = {}
    for f in glob.glob(pat):
        m = rx.match(f)
        if m:
            tiles[int(m.group(1))] = f
    if not tiles:
        return None, (1, 1)
    coords = {t: ((t - 1001) % 10, (t - 1001) // 10) for t in tiles}
    nu = max(tu for tu, _ in coords.values()) + 1
    nv = max(tv for _, tv in coords.values()) + 1
    if nu == 1 and nv == 1:
        return _load_img(next(iter(tiles.values())), max_side=max_tile), (1, 1)
    from PIL import Image
    sample = _load_img(next(iter(tiles.values())), max_side=max_tile)
    if sample is None:
        return None, (1, 1)
    tw, th = sample.size
    fill = (0, 0, 0, 255) if linear else (128, 128, 128, 255)
    atlas = Image.new("RGBA", (tw * nu, th * nv), fill)
    for t, path in tiles.items():
        tu, tv = coords[t]
        im = _load_img(path, max_side=max_tile)
        if im is None:
            continue
        if im.size != (tw, th):
            im = im.resize((tw, th))
        atlas.paste(im, (tu * tw, (nv - 1 - tv) * th))  # UDIM v up; image y down
    return atlas, (nu, nv)


def _surface_shaders(mat):
    """Bound material -> {'preview': UsdPreviewSurface shader, 'mdl': OmniPBR/MDL
    shader} (whichever exist). Robust across render contexts: scans the material's
    shader children rather than relying on a single ComputeSurfaceSource context."""
    out = {}
    for child in Usd.PrimRange(mat.GetPrim()):
        if not child.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(child)
        sid = sh.GetIdAttr().Get() if sh.GetIdAttr() else None
        if sid == "UsdPreviewSurface":
            out["preview"] = sh
        elif sh.GetSourceAsset("mdl") is not None:
            out["mdl"] = sh
    return out


def _pbr_from_prim(prim, warn):
    """Bound material -> (trimesh PBRMaterial, (nu, nv) UDIM tiling).

    Prefers UsdPreviewSurface (lossless glTF-equivalent PBR). Falls back to
    OmniPBR/MDL extraction (the Omniverse default) so Omniverse/Isaac assets keep
    their albedo + ORM + normal maps instead of degrading to flat color. Returns
    (None, (1, 1)) when no material is bound."""
    binding = UsdShade.MaterialBindingAPI(prim)
    mat = binding.ComputeBoundMaterial()[0]
    if not mat:
        return None, (1, 1)
    shaders = _surface_shaders(mat)
    if "preview" in shaders:
        return _pbr_from_preview(shaders["preview"]), (1, 1)
    if "mdl" in shaders:
        return _pbr_from_omnipbr(shaders["mdl"], warn)
    warn(f"material '{mat.GetPrim().GetName()}' has no UsdPreviewSurface or MDL "
         f"shader - geometry only")
    return None, (1, 1)


def _pbr_from_preview(shader):
    """UsdPreviewSurface -> trimesh PBRMaterial (channels are 1:1 with glTF)."""
    from trimesh.visual.material import PBRMaterial
    kw = {}
    diff = shader.GetInput("diffuseColor")
    btex = _connected_texture_file(diff)
    if btex:
        img = _load_img(btex)
        if img is not None:
            kw["baseColorTexture"] = img
    c = _const_or_none(diff)
    if c is not None:
        kw["baseColorFactor"] = [float(c[0]), float(c[1]), float(c[2]), 1.0]
    mt = _const_or_none(shader.GetInput("metallic"))
    rg = _const_or_none(shader.GetInput("roughness"))
    if mt is not None:
        kw["metallicFactor"] = float(mt)
    if rg is not None:
        kw["roughnessFactor"] = float(rg)
    mr = _pack_metallic_roughness(
        _connected_texture_file(shader.GetInput("metallic")),
        _connected_texture_file(shader.GetInput("roughness")))
    if mr is not None:
        kw["metallicRoughnessTexture"] = mr
    ntex = _load_img(_connected_texture_file(shader.GetInput("normal")))
    if ntex is not None:
        kw["normalTexture"] = ntex
    otex = _load_img(_connected_texture_file(shader.GetInput("occlusion")))
    if otex is not None:
        kw["occlusionTexture"] = otex
    em = _const_or_none(shader.GetInput("emissiveColor"))
    if em is not None:
        kw["emissiveFactor"] = [float(em[0]), float(em[1]), float(em[2])]
    etex = _load_img(_connected_texture_file(shader.GetInput("emissiveColor")))
    if etex is not None:
        kw["emissiveTexture"] = etex
    return PBRMaterial(**kw) if kw else None


def _pbr_from_omnipbr(shader, warn):
    """OmniPBR.mdl (the Omniverse default) -> (trimesh PBRMaterial, (nu, nv)).

    OmniPBR authors textures as direct asset inputs (not UsdUVTexture nodes) and
    packs occlusion/roughness/metallic into ONE ``ORM_texture`` - which is exactly
    glTF's ORM layout (R=occlusion, G=roughness, B=metallic), so it maps across
    with no recompositing. UDIM sets are atlased and the tiling is returned so the
    caller can rescale UVs."""
    from trimesh.visual.material import PBRMaterial
    kw = {}
    udim = (1, 1)

    def cval(nm):
        return _const_or_none(shader.GetInput(nm))

    # base color (albedo)
    dt = _texture_template(shader.GetInput("diffuse_texture"))
    if dt:
        img, ud = _load_udim_atlas(dt, linear=False)
        if img is not None:
            kw["baseColorTexture"] = img
            udim = ud
    dc = cval("diffuse_color_constant")
    if dc is not None:
        kw["baseColorFactor"] = [float(dc[0]), float(dc[1]), float(dc[2]), 1.0]

    # ORM packed (occlusion/roughness/metallic) -> glTF metallicRoughness + AO
    orm_on = cval("enable_ORM_texture")
    orm = _texture_template(shader.GetInput("ORM_texture"))
    if orm and (orm_on is None or bool(orm_on)):
        img, ud = _load_udim_atlas(orm, linear=True)
        if img is not None:
            kw["metallicRoughnessTexture"] = img
            kw["occlusionTexture"] = img
            if udim == (1, 1):
                udim = ud
    else:
        mr = _pack_metallic_roughness_udim(
            _texture_template(shader.GetInput("metallic_texture")),
            _texture_template(shader.GetInput("reflectionroughness_texture")))
        if mr is not None:
            img, ud = mr
            kw["metallicRoughnessTexture"] = img
            if udim == (1, 1):
                udim = ud
    mc = cval("metallic_constant")
    rc = cval("reflection_roughness_constant")
    if mc is not None:
        kw["metallicFactor"] = float(mc)
    if rc is not None:
        kw["roughnessFactor"] = float(rc)

    # normal
    nt = _texture_template(shader.GetInput("normalmap_texture"))
    if nt:
        img, ud = _load_udim_atlas(nt, linear=True)
        if img is not None:
            kw["normalTexture"] = img
            if udim == (1, 1):
                udim = ud

    # emissive
    if cval("enable_emission"):
        ec = cval("emissive_color_constant")
        if ec is not None:
            inten = cval("emissive_intensity")
            sc = float(inten) if inten else 1.0
            sc = min(sc, 1.0)  # glTF emissiveFactor is 0..1
            kw["emissiveFactor"] = [float(ec[0]) * sc, float(ec[1]) * sc,
                                    float(ec[2]) * sc]
        et = _texture_template(shader.GetInput("emissive_mask_texture"))
        if et:
            img, _ = _load_udim_atlas(et, linear=False)
            if img is not None:
                kw["emissiveTexture"] = img
    if not kw:
        return None, (1, 1)
    return PBRMaterial(**kw), udim


def _pack_metallic_roughness(metal_path, rough_path):
    """glTF packs metallic in B and roughness in G of one texture. USD preview
    surface usually authors them separately - compose them into one map."""
    if not metal_path and not rough_path:
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    mi = _load_img(metal_path)
    ri = _load_img(rough_path)
    if mi is None and ri is None:
        return None
    size = (mi or ri).size
    g = (ri.resize(size).split()[0] if ri is not None
         else Image.new("L", size, 255))
    b = (mi.resize(size).split()[0] if mi is not None
         else Image.new("L", size, 255))
    r = Image.new("L", size, 0)
    return Image.merge("RGB", (r, g, b))


def _pack_metallic_roughness_udim(metal_tmpl, rough_tmpl):
    """Like _pack_metallic_roughness but UDIM-aware (atlas each, then merge).
    Returns (image, (nu, nv)) or None."""
    if not metal_tmpl and not rough_tmpl:
        return None
    from PIL import Image
    mi, mu = _load_udim_atlas(metal_tmpl, linear=True) if metal_tmpl else (None, (1, 1))
    ri, ru = _load_udim_atlas(rough_tmpl, linear=True) if rough_tmpl else (None, (1, 1))
    if mi is None and ri is None:
        return None
    udim = ru if ri is not None else mu
    size = (ri or mi).size
    g = (ri.resize(size).split()[0] if ri is not None else Image.new("L", size, 255))
    b = (mi.resize(size).split()[0] if mi is not None else Image.new("L", size, 255))
    r = Image.new("L", size, 0)
    return Image.merge("RGB", (r, g, b)), udim


# --------------------------------------------------------------------------- #
# physics collider extraction                                                 #
# --------------------------------------------------------------------------- #
def _collider_geoms(prim, N, name, idx, meshes_dir, warn):
    """One collider prim -> list of MJCF <geom> attribute dicts (world frame).

    Honours the authored shape: native Cube/Sphere/Capsule/Cylinder gprims map to
    the matching MuJoCo primitive exactly; meshes honour MeshCollisionAPI's
    approximation (boundingCube->box, boundingSphere->sphere, convexHull/none->
    convex mesh)."""
    local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M = N @ _gf_to_np(local)
    t, quat, scale = _decompose(M)
    tp = prim.GetTypeName()
    out = []

    if tp == "Cube":
        s = UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0
        h = (s / 2.0) * scale
        out.append(dict(type="box", pos=t, quat=quat,
                        size=[h[0], h[1], h[2]]))
    elif tp == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0
        out.append(dict(type="sphere", pos=t, quat=quat,
                        size=[r * float(np.mean(scale))]))
    elif tp in ("Capsule", "Cylinder"):
        g = (UsdGeom.Capsule(prim) if tp == "Capsule" else UsdGeom.Cylinder(prim))
        r = g.GetRadiusAttr().Get() or 0.5
        hgt = g.GetHeightAttr().Get() or 1.0
        axis = g.GetAxisAttr().Get() or "Z"
        q = _qmul(quat, _axis_quat(axis))
        sca = float(np.mean(scale))
        out.append(dict(type=("capsule" if tp == "Capsule" else "cylinder"),
                        pos=t, quat=q,
                        size=[r * sca, (hgt / 2.0) * sca]))
    elif tp == "Mesh":
        approx = "none"
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none"
        res = _mesh_to_trimesh(prim, N, with_attrs=False)
        if res is None:
            return out
        tm = res[0]
        if approx == "boundingSphere":
            c = tm.bounding_sphere
            out.append(dict(type="sphere", pos=c.primitive.center,
                            quat=[1, 0, 0, 0], size=[float(c.primitive.radius)]))
        elif approx in ("boundingCube", "boundingBox"):
            ob = tm.bounding_box_oriented
            T = np.asarray(ob.primitive.transform)
            ext = np.asarray(ob.primitive.extents)
            out.append(dict(type="box", pos=T[:3, 3],
                            quat=trimesh.transformations.quaternion_from_matrix(T),
                            size=list(ext / 2.0)))
        else:  # convexHull / convexDecomposition / none -> convex mesh asset
            os.makedirs(meshes_dir, exist_ok=True)
            hull = (tm.convex_hull if approx != "none" else tm).copy()
            # Center the collision mesh at its own centre of mass and store that
            # world centre as the geom ``pos``. This makes a mesh geom behave like
            # a primitive (pos = world location, vertices local to it), so it
            # positions correctly whether emitted as a STATIC world geom OR
            # re-parented under a movable <freejoint> body: the body frame lands on
            # the COM (no lever arm) and MuJoCo computes a sane inertia from the
            # centred mesh. (A world-baked mesh at pos=0 only works for static
            # geoms and makes a movable body's COM diverge from its frame.)
            try:
                if hull.is_watertight and abs(float(hull.volume)) > 1e-9:
                    center = np.asarray(hull.center_mass, dtype=float)
                else:
                    center = np.asarray(hull.centroid, dtype=float)
            except Exception:
                center = np.asarray(hull.centroid, dtype=float)
            hull.apply_translation(-center)
            asset = f"{name}_col_{idx}"
            hull.export(os.path.join(meshes_dir, asset + ".obj"))
            out.append(dict(type="mesh", mesh=asset, pos=center,
                            quat=[1, 0, 0, 0], size=None))
    else:
        warn(f"collider prim {prim.GetPath()} type '{tp}' unsupported - skipped")
    return out


def _nearest_rigid_body(prim):
    """Walk ancestors for a RigidBodyAPI; return that prim or None (static)."""
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


# --------------------------------------------------------------------------- #
# top-level conversion                                                         #
# --------------------------------------------------------------------------- #
def convert(usd_path, name, out_dir, movable=True, fallback_obb=True, kind="auto",
            max_tile=None, floor_z=None):
    warnings: list[str] = []

    # Texture-tile size cap baked into the GLB (see _TEXTURE_MAX_TILE). <=0 = no cap.
    if max_tile is not None:
        global _TEXTURE_MAX_TILE
        _TEXTURE_MAX_TILE = int(max_tile) if int(max_tile) > 0 else (1 << 30)

    def warn(msg):
        if msg not in warnings:
            warnings.append(msg)
            print(f"  [warn] {msg}")

    os.makedirs(out_dir, exist_ok=True)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise SystemExit(f"could not open USD stage: {usd_path}")
    N = _up_axis_matrix(stage)

    # ---- gather prims -----------------------------------------------------
    # Silence pxr's C++-level material-binding warnings (a flood on Omniverse
    # scenes); the converter reports real problems via warn()/stdout, which the
    # fd-2 suppression does not touch. Set USD_TO_ASSETS_VERBOSE=1 to keep them.
    render_meshes, colliders = [], []
    with _quiet_usd():
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Imageable):
                continue
            purpose = UsdGeom.Imageable(prim).ComputePurpose()
            is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
            # visual: render/default purpose meshes that are NOT pure proxies
            if prim.IsA(UsdGeom.Mesh) and purpose != UsdGeom.Tokens.proxy \
                    and purpose != UsdGeom.Tokens.guide:
                render_meshes.append(prim)
            if is_collider:
                colliders.append(prim)

    # ---- VISUAL: build GLB (static backdrop) + collect movable visuals ----
    # A render mesh under a RigidBodyAPI is a MOVABLE object: it must NOT be baked
    # into the static GLB (it would freeze at its rest pose = a "ghost"). Instead
    # we route it through the SAME native-MuJoCo path that animates the robot --
    # it becomes a visible MuJoCo mesh inside the object's <freejoint> body, so the
    # renderer's per-frame sync_transforms moves it with physics. Only the truly
    # static meshes go into the GLB.
    glb_path = os.path.join(out_dir, f"{name}.glb")
    scene = trimesh.Scene()
    n_vis = 0
    movable_visuals: dict = {}   # rb_path(str) -> [(trimesh_world, pbr_material), ...]
    with _quiet_usd():
        for prim in render_meshes:
            rb = _nearest_rigid_body(prim) if movable else None
            res = _mesh_to_trimesh(prim, N, with_attrs=True)
            if res is None:
                continue
            tm, uv = res
            try:
                mat, udim = _pbr_from_prim(prim, warn)
            except Exception as exc:
                mat, udim = None, (1, 1)
                warn(f"material read failed on {prim.GetPath()}: {exc}")
            if uv is not None and udim != (1, 1):
                uv = uv / np.array([udim[0], udim[1]], dtype=np.float64)
            if mat is not None and uv is not None:
                tm.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
            elif mat is not None:
                tm.visual.material = mat
            if rb is not None:
                movable_visuals.setdefault(str(rb.GetPath()), []).append((tm, mat))
            else:
                scene.add_geometry(tm, node_name=prim.GetPath().name)
                n_vis += 1
    if n_vis:
        scene.export(glb_path)
    else:
        if not movable_visuals:
            warn("no render-purpose meshes found; GLB not written")
        glb_path = None   # all visuals are movable (or none) -> no static backdrop

    # ---- PHYSICS: build MJCF ---------------------------------------------
    meshes_dir = os.path.join(out_dir, f"{name}_meshes")
    static_lines, body_blocks = [], []
    CFLAGS = 'group="3" contype="1" conaffinity="1" condim="3"'
    asset_meshes = []
    gi = 0
    for prim in colliders:
        geoms = _collider_geoms(prim, N, name, gi, meshes_dir, warn)
        if not geoms:
            continue
        rb = _nearest_rigid_body(prim) if movable else None
        for g in geoms:
            gi += 1
            line = _geom_xml(name, gi, g, CFLAGS)
            if g["type"] == "mesh":
                asset_meshes.append(g["mesh"])
            if rb is not None:
                body_blocks.append((rb, prim, g, line))
            else:
                static_lines.append(line)

    have_authored = bool(static_lines or body_blocks)
    if not have_authored and fallback_obb:
        if glb_path is None:
            raise SystemExit("no authored collision and no GLB to approximate from")
        warn("no UsdPhysics colliders authored - falling back to OBB baker on the GLB")
        return _finish_obb_fallback(glb_path, name, out_dir, warnings, kind, floor_z)

    xml = _assemble_mjcf(name, static_lines, body_blocks, asset_meshes,
                         meshes_dir, movable, CFLAGS, movable_visuals)
    xml_path = os.path.join(out_dir, f"{name}_collision.xml")
    with open(xml_path, "w") as f:
        f.write(xml)
    n_vis_geoms = sum(len(v) for v in movable_visuals.values())
    print(f"USD -> visual={os.path.basename(glb_path) if glb_path else '-'}  "
          f"physics={os.path.basename(xml_path)}  "
          f"(static geoms={len(static_lines)}, movable bodies={len(body_blocks)}, "
          f"movable visuals={n_vis_geoms})")
    return glb_path, xml_path, warnings


def _geom_xml(name, gi, g, cflags):
    pos = " ".join(f"{float(x):.5f}" for x in g["pos"])
    q = " ".join(f"{float(x):.5f}" for x in g["quat"])
    if g["type"] == "mesh":
        return (f'    <geom name="{name}_col_{gi}" type="mesh" mesh="{g["mesh"]}" '
                f'pos="{pos}" quat="{q}" {cflags} rgba="0 0 0 0"/>')
    size = " ".join(f"{float(x):.5f}" for x in g["size"])
    return (f'    <geom name="{name}_col_{gi}" type="{g["type"]}" '
            f'pos="{pos}" quat="{q}" size="{size}" {cflags} rgba="0 0 0 0"/>')


def _write_obj_uv(mesh, path):
    """Write a Trimesh as OBJ with per-vertex UVs (v/vt/f). Manual writer so the
    file is exactly what MuJoCo's loader wants -- no stray .mtl/texture side files.
    Vertices are already per-corner (mesh expanded on load), so uv aligns 1:1."""
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    has_uv = uv is not None and len(uv) == len(V)
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_uv:
            for t in uv:
                f.write(f"vt {float(t[0]):.6f} {float(t[1]):.6f}\n")
            for a, b, c in (F + 1):
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
        else:
            for a, b, c in (F + 1):
                f.write(f"f {a} {b} {c}\n")


def _save_texture(img, meshes_dir, tex_registry):
    """Save a PIL image as a deduped PNG in meshes_dir. Returns basename or None."""
    if img is None:
        return None
    import io
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    data = buf.getvalue()
    digest = hashlib.md5(data).hexdigest()[:12]
    if digest not in tex_registry:
        os.makedirs(meshes_dir, exist_ok=True)
        with open(os.path.join(meshes_dir, f"tex_{digest}.png"), "wb") as f:
            f.write(data)
        tex_registry[digest] = f"tex_{digest}.png"
    return tex_registry[digest]


def _mat_props(mat):
    """trimesh PBRMaterial -> (rgba[4] in 0-1, metallic, roughness, baseColor PIL)."""
    rgba = [0.8, 0.8, 0.8, 1.0]
    metallic, roughness, tex = 0.0, 0.7, None
    if mat is not None:
        bcf = getattr(mat, "baseColorFactor", None)
        if bcf is not None:
            c = [float(x) for x in (list(bcf) + [1, 1, 1, 1])[:4]]
            if max(c[:3]) > 1.0001:          # trimesh stores as 0-255 uint8
                c = [x / 255.0 for x in c]
            rgba = c
        mf = getattr(mat, "metallicFactor", None)
        rf = getattr(mat, "roughnessFactor", None)
        if mf is not None:
            metallic = float(mf)
        if rf is not None:
            roughness = float(rf)
        tex = getattr(mat, "baseColorTexture", None)
    return rgba, metallic, roughness, tex


def _emit_visual_geoms(bname, visuals, bpos, meshes_dir,
                       mesh_rows, tex_rows, mat_rows, tex_reg, mat_reg):
    """Bake a movable body's render meshes into body-local OBJs + MuJoCo
    materials, and return visible <geom> lines (group=2, no collision, mass 0).

    These render through the SAME native-geom path that animates the robot, so the
    object's pixels track its <freejoint> physics instead of being frozen in the
    static GLB. They inherit the scene's IBL/reflections exactly like every other
    native geom."""
    rel = os.path.basename(meshes_dir)
    os.makedirs(meshes_dir, exist_ok=True)
    lines = []
    for vi, (tm, mat) in enumerate(visuals):
        local = tm.copy()
        local.apply_translation(-np.asarray(bpos, float))   # world -> body-local
        mesh_name = f"{bname}_vis{vi}"
        _write_obj_uv(local, os.path.join(meshes_dir, mesh_name + ".obj"))
        mesh_rows.append(f'    <mesh name="{mesh_name}" file="{rel}/{mesh_name}.obj"/>')

        rgba, metallic, roughness, tex_img = _mat_props(mat)
        tex_png = _save_texture(tex_img, meshes_dir, tex_reg)
        tex_attr = ""
        if tex_png:
            tex_name = f"t_{tex_png[4:-4]}"   # strip 'tex_' / '.png'
            if tex_name not in tex_reg.get("_xml", set()):
                tex_rows.append(
                    f'    <texture name="{tex_name}" type="2d" file="{rel}/{tex_png}"/>')
                tex_reg.setdefault("_xml", set()).add(tex_name)
            tex_attr = f' texture="{tex_name}"'
        mkey = (tuple(round(x, 4) for x in rgba), round(metallic, 4),
                round(roughness, 4), tex_attr)
        mat_name = mat_reg.get(mkey)
        if mat_name is None:
            mat_name = f"m_{bname}_{vi}"
            mat_reg[mkey] = mat_name
            mat_rows.append(
                f'    <material name="{mat_name}" '
                f'rgba="{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}" '
                f'metallic="{metallic:.3f}" roughness="{roughness:.3f}"{tex_attr}/>')
        lines.append(
            f'      <geom name="{mesh_name}_g" type="mesh" mesh="{mesh_name}" '
            f'material="{mat_name}" group="2" contype="0" conaffinity="0" '
            f'mass="0" pos="0 0 0" quat="1 0 0 0"/>')
    return lines


def _assemble_mjcf(name, static_lines, body_blocks, asset_meshes, meshes_dir,
                   movable, cflags, movable_visuals=None):
    movable_visuals = movable_visuals or {}
    mesh_rows = [f'    <mesh name="{m}" file="{os.path.basename(meshes_dir)}/{m}.obj"/>'
                 for m in sorted(set(asset_meshes))]
    tex_rows, mat_rows = [], []
    tex_reg, mat_reg = {}, {}

    # group rigid-body COLLISION geoms by their owning RigidBody prim
    bodies = {}
    for rb, prim, g, line in body_blocks:
        bodies.setdefault(str(rb.GetPath()), (rb, []))[1].append(g)

    body_xml = []
    handled_vis = set()
    for rbpath, (rb, geoms) in bodies.items():
        bname = f"{name}_dyn_{rb.GetName()}"
        # Body frame sits at the object's location (mean of its colliders); geoms
        # are emitted RELATIVE to it. This keeps the COM near the geometry so the
        # free body falls/settles correctly. Inertia is computed by MuJoCo from the
        # geom shapes; the authored MassAPI mass (if any) is split across the geoms
        # via the per-geom mass attribute (sum == total mass). Visual geoms carry
        # mass=0 so adding them never changes the physics.
        bpos = np.mean([np.asarray(g["pos"], float) for g in geoms], axis=0)
        per_mass = None
        if rb.HasAPI(UsdPhysics.MassAPI):
            mv = UsdPhysics.MassAPI(rb).GetMassAttr().Get()
            if mv:
                per_mass = float(mv) / len(geoms)
        glines = []
        for gi2, g in enumerate(geoms):
            rel = np.asarray(g["pos"], float) - bpos
            gg = dict(g, pos=rel)
            gl = _geom_xml(bname, f"g{gi2}", gg, cflags)
            if per_mass is not None:
                gl = gl.replace("/>", f' mass="{per_mass:.5f}"/>')
            glines.append(gl)
        # visible render geoms that track this body's freejoint
        vis = movable_visuals.get(rbpath, [])
        if vis:
            glines += _emit_visual_geoms(bname, vis, bpos, meshes_dir,
                                         mesh_rows, tex_rows, mat_rows, tex_reg, mat_reg)
            handled_vis.add(rbpath)
        bp = " ".join(f"{float(x):.5f}" for x in bpos)
        body_xml.append(
            f'    <body name="{bname}" pos="{bp}">\n'
            f'      <freejoint/>\n'
            f'{chr(10).join(glines)}\n'
            f'    </body>')

    # Movable objects that have render meshes but NO authored collider: still emit
    # a body so they're physical + visible. Use a convex hull of the visual mesh
    # for collision (cheapest reliable primitive).
    for rbpath, vis in movable_visuals.items():
        if rbpath in handled_vis or not vis:
            continue
        bname = f"{name}_dyn_{rbpath.split('/')[-1]}"
        combined = trimesh.util.concatenate([v[0] for v in vis])
        try:
            hull = combined.convex_hull
            center = (np.asarray(hull.center_mass, float)
                      if hull.is_watertight and abs(float(hull.volume)) > 1e-9
                      else np.asarray(hull.centroid, float))
        except Exception:
            center = np.asarray(combined.centroid, float)
            hull = combined
        bpos = center
        hull_local = hull.copy(); hull_local.apply_translation(-center)
        col_name = f"{bname}_colhull"
        _write_obj_uv(hull_local, os.path.join(meshes_dir, col_name + ".obj"))
        mesh_rows.append(
            f'    <mesh name="{col_name}" file="{os.path.basename(meshes_dir)}/{col_name}.obj"/>')
        glines = [f'      <geom name="{col_name}_g" type="mesh" mesh="{col_name}" '
                  f'pos="0 0 0" quat="1 0 0 0" {cflags} rgba="0 0 0 0"/>']
        glines += _emit_visual_geoms(bname, vis, bpos, meshes_dir,
                                     mesh_rows, tex_rows, mat_rows, tex_reg, mat_reg)
        bp = " ".join(f"{float(x):.5f}" for x in bpos)
        body_xml.append(
            f'    <body name="{bname}" pos="{bp}">\n'
            f'      <freejoint/>\n'
            f'{chr(10).join(glines)}\n'
            f'    </body>')

    asset_rows = mesh_rows + tex_rows + mat_rows
    asset_xml = ("  <asset>\n" + "\n".join(asset_rows) + "\n  </asset>\n") if asset_rows else ""
    body_section = ("\n".join(body_xml) + "\n") if body_xml else ""
    static_section = ("\n".join(static_lines) + "\n") if static_lines else ""
    return (
        f'<mujoco model="{name}_collision">\n'
        f'  <!-- USD-derived physics + movable visuals for the {name} scene.\n'
        f'       Static colliders are group=3 (invisible in MuJoFil/viewer); the\n'
        f'       GLB supplies the static pixels. RigidBodyAPI prims become movable\n'
        f'       <freejoint> bodies: their AUTHORED colliders (group=3, invisible)\n'
        f'       give physics, and their render meshes are emitted as VISIBLE native\n'
        f'       MuJoCo geoms (group=2, mass 0) so the renderer tracks them with the\n'
        f'       freejoint -- a pushed object moves both physically AND visually,\n'
        f'       inheriting the scene IBL/reflections like every native geom. -->\n'
        f'{asset_xml}'
        f'  <worldbody>\n'
        f'{static_section}'
        f'{body_section}'
        f'  </worldbody>\n'
        f'</mujoco>\n'
    )


def _finish_obb_fallback(glb_path, name, out_dir, warnings, kind="auto", floor_z=None):
    """Approximate collision from the GLB when the USD authored none.

    A *scene/stage* (warehouse) gets floor + walls + per-prop OBBs (the existing
    room baker). A standalone *prop* (crate, pallet, rack) gets ONE oriented box
    hugging the whole object - no floor, no walls (wrapping a 1 m crate in a room
    is nonsense). ``kind='auto'`` decides from the object's footprint."""
    M = np.eye(4)  # GLB already baked to world (Z-up, metres)
    if kind == "auto":
        mesh = trimesh.load(glb_path, force="mesh", process=False)
        ext = mesh.extents
        horiz = float(max(ext[0], ext[1]))
        # a room/stage is large and floor-like; a prop is small & compact
        kind = "scene" if horiz > 6.0 else "prop"
    if kind == "scene":
        xml_path = _obb_build_mjcf(glb_path, M, name, out_dir, floor_z=floor_z)
    else:
        xml_path = _single_obb_mjcf(glb_path, name, out_dir)
    return glb_path, xml_path, warnings


def _single_obb_mjcf(glb_path, name, out_dir):
    """One oriented bounding box hugging the whole GLB -> static prop collider.
    No floor/walls. group=3 so it stays invisible (GLB supplies the pixels)."""
    mesh = trimesh.load(glb_path, force="mesh", process=False)
    ob = mesh.bounding_box_oriented
    T = np.asarray(ob.primitive.transform)
    ext = np.asarray(ob.primitive.extents)
    pos = T[:3, 3]
    quat = trimesh.transformations.quaternion_from_matrix(T)
    cflags = 'group="3" contype="1" conaffinity="1" condim="3"'
    p = " ".join(f"{float(x):.5f}" for x in pos)
    q = " ".join(f"{float(x):.5f}" for x in quat)
    s = " ".join(f"{float(x):.5f}" for x in ext / 2.0)
    xml = (
        f'<mujoco model="{name}_collision">\n'
        f'  <!-- Single oriented-box collider hugging the {name} prop (USD authored\n'
        f'       no physics; approximated from the GLB). group=3 -> invisible; the\n'
        f'       GLB is the visual. Place/weld this where you place the GLB. -->\n'
        f'  <worldbody>\n'
        f'    <geom name="{name}_col" type="box" pos="{p}" quat="{q}" size="{s}" '
        f'{cflags} rgba="0 0 0 0"/>\n'
        f'  </worldbody>\n'
        f'</mujoco>\n'
    )
    out_path = os.path.join(out_dir, f"{name}_collision.xml")
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"prop OBB collider size(m)={np.round(ext, 3)} -> {out_path}")
    return out_path


def ensure_usd_assets(usd_path, name=None, out_dir=None, **kw):
    """Bake-once cache: (glb, xml). Re-runs only when the USD file changes."""
    name = name or os.path.splitext(os.path.basename(usd_path))[0]
    out_dir = out_dir or os.path.join(os.getcwd(), "collision")
    os.makedirs(out_dir, exist_ok=True)
    glb_path = os.path.join(out_dir, f"{name}.glb")
    xml_path = os.path.join(out_dir, f"{name}_collision.xml")
    stamp = os.path.join(out_dir, f"{name}_usd.stamp")
    key = hashlib.md5()
    key.update(str(os.path.getmtime(usd_path)).encode())
    key.update(repr(sorted(kw.items())).encode())
    digest = key.hexdigest()
    if os.path.exists(xml_path) and os.path.exists(stamp) \
            and open(stamp).read().strip() == digest:
        return glb_path, xml_path
    convert(usd_path, name, out_dir, **kw)
    with open(stamp, "w") as f:
        f.write(digest)
    return glb_path, xml_path


def main():
    ap = argparse.ArgumentParser(description="USD -> visual GLB + physics MJCF")
    ap.add_argument("usd", help="path to .usd/.usda/.usdc/.usdz")
    ap.add_argument("--name")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "collision"))
    ap.add_argument("--no-movable", dest="movable", action="store_false",
                    help="emit RigidBodyAPI prims as static instead of free bodies")
    ap.add_argument("--no-fallback", dest="fallback_obb", action="store_false",
                    help="do NOT fall back to OBB when no collision is authored")
    ap.add_argument("--kind", choices=("auto", "scene", "prop"), default="auto",
                    help="OBB-fallback shape: scene=floor+walls+props, "
                         "prop=one hugging box (auto picks by footprint)")
    ap.add_argument("--max-tile", type=int, default=1024,
                    help="cap (px) for the longest side of each baked texture tile; "
                         "raise for hero fidelity, lower for tiny GLBs, 0 = no cap "
                         "(default 1024)")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="pin the OBB-fallback ground-plane height (metres) instead "
                         "of auto-detecting; use for multi-level/mezzanine scenes "
                         "where auto-detect latches onto an upper floor")
    ap.set_defaults(movable=True, fallback_obb=True)
    args = ap.parse_args()
    name = args.name or os.path.splitext(os.path.basename(args.usd))[0]
    convert(args.usd, name, args.out, movable=args.movable,
            fallback_obb=args.fallback_obb, kind=args.kind, max_tile=args.max_tile,
            floor_z=args.floor_z)


if __name__ == "__main__":
    main()
