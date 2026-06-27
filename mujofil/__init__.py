"""mujofil: photoreal PBR rendering for GPU-resident MuJoCo.

Renders MuJoCo scenes with Google Filament (PBR + IBL) on a Vulkan device shared
with CUDA, so rendered frames are delivered straight to PyTorch as CUDA tensors
with NO CPU round-trip (zero-copy).

This package uses mujofil's renderer source (SceneBridge/materials/lights) but is
a SEPARATE build — mujofil's PyPI release is untouched.
"""
from __future__ import annotations

import atexit
import os
import sys
import weakref

try:  # single source of truth = the installed wheel's metadata (pyproject version)
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("mujofil")
except Exception:  # editable/source checkout without metadata
    __version__ = "0.2.2"

# Live renderers, closed deterministically at interpreter exit so the native
# Filament/CUDA teardown runs while the interpreter is healthy -- not during the
# unordered GC at shutdown (which can double-free and abort the process).
_LIVE_WARP_RENDERERS: "weakref.WeakSet" = weakref.WeakSet()


@atexit.register
def _close_all_warp_renderers() -> None:
    for r in list(_LIVE_WARP_RENDERERS):
        try:
            r.close()
        except Exception:
            pass

# Locate the native modules. When installed (pip), the compiled
# _mujofil_warp_gl / _mujofil_warp .so live INSIDE this package directory. In a
# dev checkout they're built into ../native by native/build*.sh — support both.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Auto-exposure target for the layered tone-map path: the 90th-percentile LINEAR
# luminance of the lit pixels is scaled to this value before the FILMIC curve, so
# highlights land just below the shoulder. Deterministic, scene-independent.
_AE_TARGET = float(os.environ.get("MUJOFIL_AE_TARGET", "1.2"))
_DEV_NATIVE = os.path.normpath(os.path.join(_HERE, os.pardir, "native"))
if os.path.isdir(_DEV_NATIVE) and _DEV_NATIVE not in sys.path:
    sys.path.insert(0, _DEV_NATIVE)

# The vendored MaterialManager loads compiled .filamat via VF_MUJOCO_MATERIALS_DIR.
# Prefer the materials bundled with this package; fall back to a mujofil install.
if "VF_MUJOCO_MATERIALS_DIR" not in os.environ:
    _pkg_materials = os.path.join(_HERE, "materials")
    if os.path.isdir(_pkg_materials):
        os.environ["VF_MUJOCO_MATERIALS_DIR"] = _pkg_materials
    else:
        try:
            import mujofil as _mujofil
            os.environ["VF_MUJOCO_MATERIALS_DIR"] = os.path.join(
                os.path.dirname(_mujofil.__file__), "materials")
        except Exception:  # noqa: BLE001
            pass

def _load_native(backend: str | None = None):
    """Select the native backend: 'gl' (default) or 'vulkan'.

    The **headless OpenGL backend is the default and the universal fallback**. It
    renders N worlds into N GL textures with a single flushAndWait and exports
    them to CUDA via GL interop, fully headless (surfaceless EGL — no X server
    required). It is the fastest and most-tested path.

    The Vulkan backend ('vulkan'/'vk') is optional/experimental. If it is
    requested but cannot be loaded or initialised on this machine, mujofil falls
    back to the headless OpenGL backend and emits a clear warning. Override with
    MUJOFIL_BACKEND=gl|vulkan.
    """
    # MUJOFIL_BACKEND is the current name; MUJOFIL_WARP_BACKEND is still honoured
    # for backward compatibility with code written against the 0.1.x package.
    env = os.environ.get("MUJOFIL_BACKEND") or os.environ.get("MUJOFIL_WARP_BACKEND")
    backend = (backend or env or "gl").lower()

    def _load_gl():
        import _mujofil_warp_gl as _gl  # noqa: E402  (headless EGL — the default)
        return _gl

    if backend in ("gl", "opengl"):
        return _load_gl()

    if backend in ("vulkan", "vk"):
        try:
            import _mujofil_warp as _vk  # noqa: E402
            return _vk
        except Exception as exc:
            # Vulkan is optional/experimental; the headless OpenGL backend is the
            # universal fallback. Degrade gracefully with a clear message rather
            # than crashing on an import/init error.
            import warnings
            warnings.warn(
                "MUJOFIL_BACKEND=vulkan could not be loaded "
                f"({type(exc).__name__}: {exc}); falling back to the headless "
                "OpenGL backend (the default).",
                RuntimeWarning,
                stacklevel=2,
            )
            return _load_gl()

    raise ValueError(f"unknown MUJOFIL_BACKEND={backend!r} (use 'gl' or 'vulkan')")


# The native renderer module (Filament + CUDA + EGL/Vulkan) is loaded LAZILY on
# first use, not at import time. This keeps `import mujofil` -- and in
# particular the CPU-only asset tools (`mujofil.tools.*`) -- from spinning up
# the GPU renderer (which also avoids a libpng/zlib symbol clash between the
# native .so and trimesh's GLB exporter during USD->GLB conversion).
_native = None


def _get_native():
    global _native
    if _native is None:
        # Import torch BEFORE loading the native (cudart-linked) .so. On a
        # torch/driver CUDA mismatch, torch's lazy CUDA probe
        # (torch.cuda.is_available()) SEGFAULTS if torch's C extension is first
        # imported *after* the native module has loaded the CUDA runtime. Loading
        # torch first makes that probe fail gracefully (returns False) so the
        # preflight can report a clear, actionable error instead of crashing.
        # torch is a hard dependency of this package, so importing it here is safe.
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001 - missing torch is reported by _preflight_check
            pass
        _native = _load_native()
    return _native


def _preflight_check() -> None:
    """Fail with a CLEAR, actionable message -- not a raw native crash -- when the
    machine is missing what mujofil needs: an NVIDIA GPU (the gl_Layer /
    CUDA-interop path is NVIDIA-only) and a CUDA-enabled PyTorch (the renderer
    delivers frames as zero-copy torch.cuda tensors). Runs once, before the native
    renderer is constructed. Set MUJOFIL_SKIP_PREFLIGHT=1 to bypass.
    """
    global _preflight_done
    if (_preflight_done
            or os.environ.get("MUJOFIL_SKIP_PREFLIGHT") == "1"
            or os.environ.get("MUJOFIL_WARP_SKIP_PREFLIGHT") == "1"):
        return

    problems = []

    # 1. PyTorch with CUDA -- the package's whole value is zero-copy torch.cuda.
    try:
        import torch  # noqa: F401
        try:
            if not torch.cuda.is_available():
                problems.append(
                    "PyTorch is installed but reports CUDA is NOT available. "
                    "mujofil delivers frames as zero-copy CUDA tensors and "
                    "needs a CUDA-enabled PyTorch on an NVIDIA GPU.\n"
                    "  • Check `nvidia-smi` works and your driver is installed.\n"
                    "  • Install a torch build matching your GPU's CUDA arch "
                    "(e.g. cu124, or cu128 for RTX 50xx/Blackwell): "
                    "https://pytorch.org/get-started/locally/")
        except Exception:
            pass  # torch present but cuda probe failed oddly -- don't block here
    except ImportError:
        problems.append(
            "PyTorch is NOT installed. mujofil returns rendered frames as "
            "zero-copy torch.cuda tensors, so it needs PyTorch.\n"
            "  • Install:  pip install \"mujofil[torch]\"   (or install a "
            "torch build for your GPU's CUDA arch from "
            "https://pytorch.org/get-started/locally/)")

    # 2. NVIDIA GPU presence (the gl_Layer parallel-batch path is NVIDIA-only).
    #    Use a lightweight check that doesn't import torch a second time.
    import ctypes.util
    has_nvidia = False
    try:
        import torch  # noqa: F811
        has_nvidia = bool(getattr(torch.version, "cuda", None)) and torch.cuda.is_available()
    except Exception:
        has_nvidia = False
    if not has_nvidia:
        # Secondary signal: the NVIDIA management/driver library being loadable.
        if (ctypes.util.find_library("nvidia-ml") or
                ctypes.util.find_library("cuda") or
                os.path.exists("/proc/driver/nvidia/version")):
            has_nvidia = True
    if not has_nvidia and not any("PyTorch is NOT installed" in p for p in problems):
        problems.append(
            "No NVIDIA GPU/driver detected. mujofil's parallel-batch "
            "(gl_Layer) renderer and CUDA zero-copy require an NVIDIA GPU on "
            "Linux x86_64.\n"
            "  • Verify with `nvidia-smi`. Headless servers still need the "
            "NVIDIA driver installed.")

    if problems:
        msg = ("mujofil cannot run on this machine:\n\n" +
               "\n".join("  " + p for p in problems) +
               "\n\n(Set MUJOFIL_SKIP_PREFLIGHT=1 to bypass this check.)")
        raise RuntimeError(msg)

    _preflight_done = True


_preflight_done = False


def __getattr__(name):  # PEP 562: lazy module attribute
    if name == "RendererConfig":
        return _get_native().RendererConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_config(
    *,
    width: int = 256,
    height: int = 256,
    batch_size: int = 1,
    # --- quality toggles: turn OFF to trade fidelity for throughput ---
    ssao: bool = True,
    ssao_quality: str | int = "ultra",
    ssao_ssct: bool = True,
    shadows: bool = True,
    msaa: bool = True,
    msaa_samples: int = 4,
    bloom: bool = False,
    fxaa: bool = False,
    ssr: bool = True,
    # --- tone mapping / exposure ---
    exposure: float = 0.0,
    tone_mapping: bool = True,
    dithering: bool = True,
    layered: bool = False,
) -> "RendererConfig":
    """Build a :class:`RendererConfig` from clear keyword toggles.

    Quality toggles (each can be flipped independently to reproduce the
    fidelity/throughput trade-offs in ``benchmarks/``):

    - ``ssao``          screen-space ambient occlusion. **Biggest single cost** --
                        turning it off is ~2x faster (the ``fast`` preset).
    - ``ssao_quality``  SSAO quality: "low" | "medium" | "high" | "ultra" (or 0-3).
                        Affects the look more than the speed -- the SSAO *pass*
                        dominates, not its quality level.
    - ``ssao_ssct``     SSAO screen-space cone tracing (extra contact shadows).
                        A small extra cost on top of SSAO.
    - ``shadows``       soft shadow maps.
    - ``msaa`` / ``msaa_samples``  multi-sample anti-aliasing (2/4/8).
    - ``bloom``         HDR bloom (off by default; cheap-ish).
    - ``fxaa``          fast approximate AA (alternative to MSAA).
    - ``ssr``           screen-space reflections on glossy surfaces (e.g. a
                        polished floor mirrors nearby geometry).
    - ``exposure``      linear exposure multiplier before tone mapping.
    - ``tone_mapping``  FILMIC tone mapping (vs LINEAR when False).
    - ``dithering``     temporal dithering to avoid banding.

    ``batch_size`` must be >= the number of worlds passed to ``render_batch``.
    """
    _ssao_levels = {"low": 0, "medium": 1, "high": 2, "ultra": 3}
    if isinstance(ssao_quality, str):
        try:
            ssao_q = _ssao_levels[ssao_quality.lower()]
        except KeyError:
            raise ValueError(
                f"ssao_quality must be one of {list(_ssao_levels)} or 0-3, "
                f"got {ssao_quality!r}")
    else:
        ssao_q = max(0, min(3, int(ssao_quality)))
    cfg = _get_native().RendererConfig()
    cfg.width = width
    cfg.height = height
    cfg.batch_size = batch_size
    cfg.layered = layered
    cfg.enable_ssao = ssao
    cfg.ssao_quality = ssao_q
    cfg.ssao_ssct = ssao_ssct
    cfg.enable_shadows = shadows
    cfg.enable_msaa = msaa
    cfg.msaa_samples = msaa_samples
    cfg.enable_bloom = bloom
    cfg.enable_fxaa = fxaa
    cfg.enable_ssr = ssr
    cfg.exposure = exposure
    cfg.tone_mapping = tone_mapping
    cfg.dithering = dithering
    return cfg


# Named quality presets so users can reproduce our benchmark trends on their own
# hardware. Pass to ``WarpRenderer(preset=...)``; override individual toggles too.
#
# THE TWO HEADLINE PRESETS (the ones to use):
#   eval   -- full photoreal fidelity (ULTRA SSAO + cone tracing + 4x MSAA +
#             shadows + screen-space reflections). Use for evaluation videos,
#             cinematics, sim-to-real demos -- anything a HUMAN watches.
#   train  -- throughput-optimised observations for batched vision-RL. The
#             per-view screen-space passes -- SSAO, SSR (reflections) and 4x MSAA
#             -- are OFF; full PBR materials, textures, IBL, FILMIC tone mapping
#             and shadows are KEPT. Those screen-space passes run once PER VIEW
#             (N times in a batch) and are imperceptible to a low-res conv policy,
#             so dropping them is pure throughput. Measured ~4.7x faster than
#             ``eval`` at 128px N=64 (RTX 4060, photoreal cafe 1442 -> 6742 cam/s)
#             with the materials/lighting unchanged -- the image stays photoreal,
#             not flat. SSR was the dominant hidden cost.
#
# Finer-grained presets (kept for reproducing the benchmark trade-off curve):
#   high == eval; fast/medium/raw are intermediate points; ultra adds 8x MSAA+bloom.
QUALITY_PRESETS = {
    # --- the two you should reach for ---
    "eval": dict(ssao=True, ssao_quality="ultra", ssao_ssct=True,
                 shadows=True, msaa=True, msaa_samples=4, ssr=True),
    "train": dict(ssao=False, ssr=False, shadows=True, msaa=False),
    # --- intermediate / benchmark points ---
    "high": dict(ssao=True, ssao_quality="ultra", ssao_ssct=True,
                 shadows=True, msaa=True, msaa_samples=4, ssr=True),
    "medium": dict(ssao=True, ssao_quality="high", ssao_ssct=False,
                   shadows=True, msaa=True, msaa_samples=4, ssr=False),
    "fast": dict(ssao=False, ssr=False, shadows=True, msaa=True, msaa_samples=4),
    "ultra": dict(ssao=True, ssao_quality="ultra", ssao_ssct=True,
                  shadows=True, msaa=True, msaa_samples=8, bloom=True, ssr=True),
    "raw": dict(ssao=False, ssr=False, shadows=False, msaa=False),
}


class WarpRenderer:
    """PBR renderer that returns frames as torch CUDA tensors (zero-copy).

    Construct it three ways::

        # 1. keyword toggles (recommended)
        r = WarpRenderer(width=256, height=256, batch_size=32, ssao=False)

        # 2. a named quality preset ("train" | "eval", or "fast"/"raw"/...)
        r = WarpRenderer(width=256, batch_size=32, preset="train")

        # 3. an explicit RendererConfig
        r = WarpRenderer(config=make_config(width=256, ssao=True))

    Then::

        r.load_model(model)                # mujoco.MjModel
        r.sync_camera(model, data, cam_id=0)
        img = r.render()                   # (H, W, 4) uint8 torch.cuda tensor

    Quality toggles (``ssao``, ``shadows``, ``msaa``, ``bloom``, ``fxaa``,
    ``exposure``, ``tone_mapping``, ``dithering``) are documented on
    :func:`make_config`. ``ssao`` is the biggest throughput lever (~2x off).

    Two headline presets cover the two real use-cases:
      * ``preset="eval"``  -- full photoreal fidelity (cinematics, evaluation).
      * ``preset="train"`` -- ~2x faster batched observations for vision-RL,
        with no impact on what a CNN policy learns.
    """

    def __init__(self, config: "RendererConfig | None" = None,
                 *, preset: str | None = None, **toggles):
        # Clear, actionable error (not a raw native abort/segfault) if torch/NVIDIA
        # are missing. Run this BEFORE make_config(), which loads the native
        # cudart-linked module: importing torch *after* that module makes torch's
        # CUDA probe segfault on a driver/runtime mismatch, defeating the whole
        # point of the preflight (a clean message instead of a crash).
        _preflight_check()
        if config is None:
            kw = dict(QUALITY_PRESETS[preset]) if preset else {}
            kw.update(toggles)
            config = make_config(**kw)
        elif preset or toggles:
            raise TypeError("pass either `config=` or keyword toggles/`preset=`, not both")
        native = _get_native()
        # The layered (single-draw parallel-batch) path is OpenGL-only: it relies
        # on the gl_Layer routing fork that exists solely in the GL backend. The
        # Vulkan backend only carries link-time stubs that throw, and pybind11
        # reports that C++ exception as an opaque "unknown exception", so guard it
        # HERE with a clear, actionable message before touching the native module.
        if getattr(config, "layered", False):
            if getattr(native, "__name__", "").endswith("_warp"):  # the Vulkan module
                raise ValueError(
                    "layered rendering is only available on the OpenGL backend, "
                    "but MUJOFIL_BACKEND=vulkan is selected. Use the default "
                    "OpenGL backend (unset MUJOFIL_BACKEND, or set it to 'gl') "
                    "for layered rendering; the Vulkan backend uses the "
                    "per-frame batch path (render_batch).")
            # Layered needs the gl_Layer material set, compiled separately
            # (matc -g). Point the material loader at it BEFORE the native
            # renderer builds its MaterialManager.
            _lm = os.path.join(_HERE, "materials_layered")
            if os.path.isdir(_lm):
                os.environ["VF_MUJOCO_MATERIALS_DIR"] = _lm
        self._r = native.WarpRenderer(config)
        # Capture the camera exposure (EV) so the layered compositor can apply the
        # SAME deterministic FILMIC+exposure tonemap Filament's post-processing
        # applies in the render_batch path (the layered OBJECTS pass runs with
        # post-processing OFF to preserve gl_Layer routing, so we replicate it).
        self._exposure_ev = float(getattr(config, "exposure", 0.0) or 0.0)
        # True once a GLB environment is ingested into the layered objects pass
        # (load_glb_layered) -> the layered compositor tonemaps it deterministically.
        self._has_layered_env = False
        _LIVE_WARP_RENDERERS.add(self)

    def close(self) -> None:
        """Release the native renderer (Filament + CUDA). Idempotent; safe to
        call, and safe NOT to call (an atexit handler closes any that remain in a
        deterministic order, avoiding the GC-at-shutdown teardown crash)."""
        r = self.__dict__.pop("_r", None)
        del r  # drop the only Python ref -> native dtor runs now, not at GC

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- scene ---
    def load_model(self, model):
        self._r.load_model(_addr(model))

    def load_glb(self, path: str):
        self._r.load_glb(path)

    def load_glb_xform(self, path: str, xform16):
        """Load a GLB backdrop placed by a column-major 4x4 transform (16 floats).
        Use the same xform the rest of the pipeline uses (e.g. to orient a Y-up
        scene into the MuJoCo Z-up world with the floor at z~0); without it the
        GLB sits in its raw authored space and won't align with the MuJoCo geoms.
        """
        xs = [float(x) for x in (xform16.flatten() if hasattr(xform16, "flatten")
                                 else xform16)]
        if len(xs) != 16:
            raise ValueError("xform16 must have 16 elements (column-major 4x4)")
        self._r.load_glb_xform(path, xs)

    def load_glb_layered(self, path: str, xform16=None, max_tex: int = 1024):
        """Ingest a GLB ENVIRONMENT into the INSTANCED LAYERED path so it works
        with per-world EGOCENTRIC cameras in the single parallel draw (unlike
        ``load_glb*`` which adds the GLB as one shared, non-instanced backdrop that
        can only be rendered from a shared camera).

        The GLB is parsed (trimesh), each mesh baked into world space by ``xform16``
        (column-major 4x4, same convention as ``load_glb_xform``; identity if None),
        and handed to the native instanced layered pipeline with its base colour +
        albedo texture. Then ``render_batch_layered(cam_id=ego)`` renders every
        world's egocentric view of the FULL environment in ONE instanced draw.

        Fidelity note: only base colour + albedo map are carried over (the layered
        material set has no normal/MR/emissive maps); this is an RL-observation
        grade ingest, not a hero-shot path. Requires ``layered=True``.
        """
        import numpy as np
        try:
            import trimesh
        except Exception as e:
            raise RuntimeError("load_glb_layered needs trimesh (pip install trimesh)") from e
        if not getattr(self._r, "layered", False):
            raise RuntimeError("load_glb_layered requires a renderer built with layered=True")

        # column-major 4x4 -> row-major numpy for point transforms
        if xform16 is None:
            M = np.eye(4, dtype=np.float64)
        else:
            xs = np.asarray(xform16, dtype=np.float64).reshape(-1)
            if xs.size != 16:
                raise ValueError("xform16 must have 16 elements (column-major 4x4)")
            M = xs.reshape(4, 4, order="F")     # column-major -> (row,col)
        R = M[:3, :3]

        scene = trimesh.load(path, process=False, force="scene")
        # dump(concatenate=False) bakes each mesh's node transform into its verts
        meshes = scene.dump(concatenate=False) if hasattr(scene, "dump") else [scene]
        n_added = 0

        def _tex_bytes(img, max_px):
            """PIL/ndarray glTF texture -> (rgba_bytes, w, h) or (b'', 0, 0)."""
            if img is None:
                return b"", 0, 0
            from PIL import Image as _Im
            im = img if isinstance(img, _Im.Image) else _Im.fromarray(np.asarray(img))
            im = im.convert("RGBA")
            if max(im.size) > max_px:
                s = max_px / max(im.size)
                im = im.resize((max(1, int(im.size[0]*s)), max(1, int(im.size[1]*s))))
            arr = np.asarray(im, dtype=np.uint8)
            return arr.tobytes(), int(arr.shape[1]), int(arr.shape[0])

        def _same_image(a, b):
            """True if two glTF textures are the same image (exporter dup detect)."""
            try:
                aa = np.asarray(a.convert("RGB") if hasattr(a, "convert") else a)
                bb = np.asarray(b.convert("RGB") if hasattr(b, "convert") else b)
                return aa.shape == bb.shape and np.array_equal(aa, bb)
            except Exception:
                return False

        def _uv_tangents(v, faces, uv, nrm):
            """Per-vertex UV-aligned tangent (T.xyz, handedness) for normal maps.
            Standard Lengyel accumulation, vectorised."""
            tan1 = np.zeros((len(v), 3), np.float64)
            tan2 = np.zeros((len(v), 3), np.float64)
            f = faces.reshape(-1, 3)
            i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
            e1 = v[i1] - v[i0]; e2 = v[i2] - v[i0]
            du1 = uv[i1] - uv[i0]; du2 = uv[i2] - uv[i0]
            denom = (du1[:, 0]*du2[:, 1] - du2[:, 0]*du1[:, 1])
            rinv = np.where(np.abs(denom) > 1e-12, 1.0/np.where(denom == 0, 1, denom), 0.0)
            sdir = (e1*du2[:, 1:2] - e2*du1[:, 1:2]) * rinv[:, None]
            tdir = (e2*du1[:, 0:1] - e1*du2[:, 0:1]) * rinv[:, None]
            for arr, src in ((tan1, sdir), (tan2, tdir)):
                np.add.at(arr, i0, src); np.add.at(arr, i1, src); np.add.at(arr, i2, src)
            n = nrm
            ndt = np.einsum("ij,ij->i", n, tan1)
            t = tan1 - n * ndt[:, None]
            tl = np.linalg.norm(t, axis=1, keepdims=True)
            t = np.where(tl > 1e-8, t / np.where(tl == 0, 1, tl), np.array([1.0, 0, 0]))
            # handedness
            cross = np.cross(n, tan1)
            w = np.where(np.einsum("ij,ij->i", cross, tan2) < 0.0, -1.0, 1.0)
            return np.concatenate([t, w[:, None]], axis=1).astype(np.float32)

        for mesh in meshes:
            if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                continue
            v = np.asarray(mesh.vertices, dtype=np.float64)
            # to world: apply the scene xform (mesh node transform already baked)
            vw = (v @ R.T) + M[:3, 3]
            if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(v):
                nw = np.asarray(mesh.vertex_normals, dtype=np.float64) @ R.T
                nl = np.linalg.norm(nw, axis=1, keepdims=True)
                nw = np.where(nl > 1e-9, nw / np.where(nl == 0, 1, nl), np.array([0.0, 0, 1.0]))
            else:
                nw = np.zeros_like(vw); nw[:, 2] = 1.0
            faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)

            # material: base colour factor + glTF map set (albedo/normal/MR/emissive)
            r = g = b = a = 1.0
            rough, metal = 1.0, 1.0       # factors; modulated by the MR map if present
            er = eg = eb = 0.0
            uvs = None
            alb = nrm_b = mr_b = em_b = b""
            aw = ah = nwid = nh = mw = mh = ew = eh = 0
            vis = getattr(mesh, "visual", None)
            mat = getattr(vis, "material", None)
            uv = getattr(vis, "uv", None)
            has_uv = uv is not None and len(uv) == len(v)
            if mat is not None:
                bcf = getattr(mat, "baseColorFactor", None)
                if bcf is not None:
                    c = np.asarray(bcf, dtype=np.float64).reshape(-1) / (
                        255.0 if np.asarray(bcf).max() > 1.5 else 1.0)
                    r, g, b = float(c[0]), float(c[1]), float(c[2])
                    a = float(c[3]) if c.size > 3 else 1.0
                rf = getattr(mat, "roughnessFactor", None)
                mf = getattr(mat, "metallicFactor", None)
                if rf is not None:
                    rough = float(rf)
                if mf is not None:
                    metal = float(mf)
                ef = getattr(mat, "emissiveFactor", None)
                if ef is not None:
                    ec = np.asarray(ef, dtype=np.float64).reshape(-1)
                    er, eg, eb = float(ec[0]), float(ec[1]), float(ec[2])
                if has_uv:
                    alb, aw, ah = _tex_bytes(
                        getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None),
                        max_tex)
                    nrm_b, nwid, nh = _tex_bytes(getattr(mat, "normalTexture", None), max_tex)
                    mr_b, mw, mh = _tex_bytes(getattr(mat, "metallicRoughnessTexture", None), max_tex)
                    # glTF: emissive = emissiveFactor * emissiveTexture. A zero
                    # factor means NO emission regardless of the texture, so only
                    # sample the map when the factor is non-zero (else surfaces
                    # would wrongly glow at full strength).
                    if er > 0.0 or eg > 0.0 or eb > 0.0:
                        etex = getattr(mat, "emissiveTexture", None)
                        # Robustness: some exporters DUPLICATE the base-colour map
                        # into the emissive slot with emissiveFactor=[1,1,1]. That
                        # makes every surface emit its own albedo (doubled, posterised
                        # glow -- e.g. a floor "glowing"). When the emissive map is
                        # the same image object/bytes as the albedo map, treat it as
                        # an export artifact and drop the emissive (a genuinely
                        # emissive surface uses a DISTINCT map).
                        base_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
                        if etex is not None and base_img is not None and (
                                etex is base_img or _same_image(etex, base_img)):
                            etex = None
                        if etex is not None:
                            em_b, ew, eh = _tex_bytes(etex, max_tex)
            if has_uv:
                uvv = np.asarray(uv, dtype=np.float32).copy()
                uvv[:, 1] = 1.0 - uvv[:, 1]          # glTF/Filament V flip
                uvs = uvv.reshape(-1).tolist()
                # UV-aligned tangents (only needed when a normal map is present)
                if nrm_b:
                    tang = _uv_tangents(vw, faces, uvv.astype(np.float64), nw)
                    tang_list = tang.reshape(-1).tolist()
                else:
                    tang_list = []
            else:
                uvs = []
                tang_list = []

            self._r.add_layered_env_mesh(
                vw.astype(np.float32).reshape(-1).tolist(),
                nw.astype(np.float32).reshape(-1).tolist(),
                uvs, tang_list,
                faces.tolist(),
                float(r), float(g), float(b), float(a),
                float(rough), float(metal),
                float(er), float(eg), float(eb),
                alb, int(aw), int(ah),
                nrm_b, int(nwid), int(nh),
                mr_b, int(mw), int(mh),
                em_b, int(ew), int(eh))
            n_added += 1
        if n_added:
            self._has_layered_env = True
        return n_added

    def load_ibl(self, ibl_ktx: str, skybox_ktx: str):
        self._r.load_ibl(ibl_ktx, skybox_ktx)

    def set_ambient_intensity(self, intensity: float):
        self._r.set_ambient_intensity(intensity)

    def clear_dynamic_lights(self):
        self._r.clear_dynamic_lights()

    def add_directional_light(self, dx, dy, dz, r, g, b, intensity, cast_shadows=False):
        self._r.add_directional_light(dx, dy, dz, r, g, b, intensity, cast_shadows)

    def add_point_light(self, x, y, z, r, g, b, intensity, falloff):
        self._r.add_point_light(x, y, z, r, g, b, intensity, falloff)

    def add_spot_light(self, x, y, z, dx, dy, dz, r, g, b, intensity, falloff,
                       inner_deg, outer_deg, focused=True):
        self._r.add_spot_light(x, y, z, dx, dy, dz, r, g, b, intensity, falloff,
                               inner_deg, outer_deg, focused)

    def set_free_camera(self, ex, ey, ez, tx, ty, tz):
        self._r.set_free_camera(ex, ey, ez, tx, ty, tz)

    # --- per-frame ---
    def sync_transforms(self, model, data):
        self._r.sync_transforms(_addr(model), _addr(data))

    def sync_camera(self, model, data, cam_id: int = -1):
        _check_cam(model, cam_id)
        self._r.sync_camera(_addr(model), _addr(data), cam_id)

    def render(self):
        """Render and return an (H, W, 4) uint8 torch.cuda tensor."""
        import torch
        # Filament/OpenGL read back with a BOTTOM-left origin, so the raw buffer
        # is vertically flipped vs MuJoCo / standard image convention (row 0 =
        # top). Flip the height axis so callers get an upright image that matches
        # mujoco.Renderer. (GPU flip; cheap.)
        return torch.from_dlpack(self._r.render_dlpack()).flip(0)

    def render_batch(self, model, datas, cam_id: int = -1):
        """Render N worlds (one MjData each) with ONE GPU sync.

        Returns an (N, H, W, 4) uint8 torch.cuda tensor. The renderer must have
        been created with config.batch_size >= len(datas).
        """
        import torch
        _check_cam(model, cam_id)
        ptrs = [_addr(d) for d in datas]
        # Flip H (axis 1) to upright -- see render() for why.
        return torch.from_dlpack(
            self._r.render_batch_dlpack(_addr(model), ptrs, cam_id)).flip(1)

    def render_batch_layered(self, model, datas, cam_id: int = -1,
                             tonemap=None, exposure=None):
        """LAYERED parallel batch: render N worlds in ONE instanced GPU pass.

        Requires a renderer built with ``layered=True`` (and ``batch_size >= N``).
        All N worlds render in a single draw via the forked Filament gl_Layer path
        (NVIDIA only); per-world transforms live GPU-side in InstanceBuffers, so
        there is no per-world CPU render loop. Returns (N, H, W, 4) uint8 torch.cuda.

        Camera:
          * ``cam_id < 0`` (default): a SINGLE shared camera for all worlds
            (fixed/overhead view) -- the original byte-identical path.
          * ``cam_id >= 0``: EGOCENTRIC -- every world renders from its OWN copy of
            that MuJoCo camera (e.g. a robot-mounted camera that moves with each
            world's state), still in ONE instanced draw. Implemented by folding each
            world's view matrix into that world's per-instance transform and using a
            shared projection, so geometry and depth are correct per world. Works for
            NATIVE MJCF geometry AND for photoreal GLB environments ingested via
            ``load_glb_layered`` (the GLB becomes instanced layered geometry).

        Tonemapping (deterministic, no per-scene tuning):
          The layered OBJECTS pass runs with post-processing OFF (to preserve the
          gl_Layer routing), so its colour is LINEAR. When the environment is in
          that pass (egocentric, or a GLB ingested via ``load_glb_layered``) we
          apply the SAME tone mapping Filament's post-processing applies in the
          ``render_batch`` path -- exposure (``2^EV``) then the FILMIC/ACES curve
          then sRGB encode -- driven by the renderer's ``config.exposure`` (EV).
          This is fully deterministic and identical across all scenes; set
          ``config.exposure`` once like any renderer. ``tonemap``/``exposure`` here
          are optional manual overrides (``exposure`` in EV stops); leave them None
          for the automatic, shippable behaviour. The plain shared-camera path with
          no ingested env stays byte-identical (tonemap off).
        """
        import torch
        if not self._r.layered:
            raise RuntimeError("render_batch_layered requires a renderer built with "
                               "layered=True (e.g. WarpRenderer(layered=True, batch_size=N))")
        _check_cam(model, cam_id)
        ptrs = [_addr(d) for d in datas]
        obj = torch.from_dlpack(self._r.render_batch_layered_dlpack(_addr(model), ptrs, cam_id))
        # `obj` is (N,H,W,4) FLOAT16: linear-HDR RGB + 0..1 coverage alpha (the
        # objects pass renders to a half-float RT so dark regions keep full
        # precision for the tone map). Composite over the shared static backdrop
        # (GLB + skybox), an already-tonemapped RGBA8 display image broadcast to
        # every world. Single GPU blend, no CPU bounce.
        bg = torch.from_dlpack(self._r.layered_backdrop_dlpack())   # (H,W,4) uint8
        a = obj[..., 3:4].to(torch.float32)                         # (N,H,W,1) 0..1

        # Tonemap the LINEAR objects-pass colour exactly as Filament does, when the
        # environment lives in that pass (egocentric, or an ingested layered GLB).
        # The plain shared-camera path over an already-tonemapped backdrop leaves
        # objects untonemapped (small manipulables) so that path is unchanged.
        if tonemap is None:
            tonemap = (cam_id >= 0) or getattr(self, "_has_layered_env", False)
        if tonemap:
            # Replicate Filament's post-processing tone map (which the layered
            # OBJECTS pass runs WITHOUT) so the egocentric / ingested-env output is
            # display-referred and DETERMINISTIC -- no per-scene magic numbers.
            #
            # Pipeline: robust auto-exposure -> FILMIC/ACES (Filament's curve, on
            # luminance) -> sRGB. The auto-exposure exposes for the HIGHLIGHTS (a
            # high luminance percentile mapped just below the FILMIC shoulder),
            # which is robust to the bimodal indoor histogram (bright walls + dark
            # corners): bright scenes don't blow out and dim, enclosed scenes get
            # lifted. It is a pure, deterministic function of the pixels, so a
            # brand-new environment is exposed correctly zero-shot. ``config.exposure``
            # (EV) biases it; ``exposure`` here is an optional manual EV override.
            # `obj` is FLOAT16 linear HDR (no decode, no quantization) -- that is
            # what lets the tone map lift darks WITHOUT posterisation.
            x = obj[..., :3].to(torch.float32)            # linear HDR
            if exposure is None:
                lum = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])
                Nw = lum.shape[0]
                flat = lum.reshape(Nw, -1)
                amask = (a[..., 0].reshape(Nw, -1) > 0.004)
                # Per-world 90th-percentile of the LIT pixels, computed for ALL N
                # worlds in ONE vectorised reduction. The earlier per-world Python
                # loop launched N separate quantile kernels + N boolean-mask
                # allocations per frame, which dominated the whole step at large N
                # (~240ms @N=1024); this is the same result in a few ms.
                #
                # We subsample to a fixed pixel budget per world: auto-exposure only
                # needs a representative percentile, a few thousand pixels give a
                # stable estimate, and it keeps the batched tensor under torch's
                # quantile element cap (2**24) at any N/resolution.
                P = flat.shape[1]
                budget = 4096
                if P > budget:
                    idx = torch.linspace(0, P - 1, budget, device=flat.device).long()
                    flat_s = flat.index_select(1, idx)
                    amask_s = amask.index_select(1, idx)
                else:
                    flat_s, amask_s = flat, amask
                masked = torch.where(amask_s, flat_s,
                                     torch.tensor(float("nan"), device=flat_s.device))
                p_lit = torch.nanquantile(masked, 0.90, dim=1)
                p_all = torch.quantile(flat_s, 0.90, dim=1)
                enough = amask_s.sum(dim=1) >= 64
                p = torch.where(enough, p_lit, p_all).clamp_min(1e-4)
                scale = (_AE_TARGET / p).clamp(0.05, 256.0)
                scale = scale * (2.0 ** self._exposure_ev)
                x = x * scale.view(-1, 1, 1, 1)
            else:
                ev = float(exposure)
                if ev != 0.0:
                    x = x * (2.0 ** ev)                  # Filament: v * exp2(EV)
            # FILMIC / ACES (Narkowicz) on LUMINANCE with chroma preserved (ratio-
            # scale RGB by tonemapped/linear luminance) -- avoids the per-channel
            # hue shift that FILMIC's independent division causes in dark regions.
            a_, b_, c_, d_, e_ = 2.51, 0.03, 2.43, 0.59, 0.14
            lum = (0.2126 * x[..., 0:1] + 0.7152 * x[..., 1:2] + 0.0722 * x[..., 2:3])
            lum = lum.clamp_min(1e-6)
            lout = ((lum * (a_ * lum + b_)) / (lum * (c_ * lum + d_) + e_)).clamp_(0.0, 1.0)
            x = (x * (lout / lum)).clamp_(0.0, 1.0)
            # linear -> sRGB (display encode)
            x = torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)
            obj_rgb = (x * 255.0)
        else:
            # Untonemapped path: scale linear [0,1] objects to display range as the
            # 8-bit-linear RT used to (small manipulables over the backdrop).
            obj_rgb = (obj[..., :3].to(torch.float32) * 255.0)
        out = obj_rgb * a + bg[..., :3].to(torch.float32) * (1.0 - a)
        rgba = torch.empty((obj.shape[0], obj.shape[1], obj.shape[2], 4),
                           dtype=torch.uint8, device=obj.device)
        rgba[..., :3] = out.clamp_(0.0, 255.0).to(torch.uint8)
        rgba[..., 3] = 255
        # Flip H (axis 1) to upright -- Filament/GL read back bottom-up; see
        # render(). obj + bg share that origin so one flip of the composite is
        # correct for both.
        return rgba.flip(1)

    @property
    def layered(self) -> bool:
        return self._r.layered

    def reset_profile(self):
        self._r.reset_profile()

    def profile(self) -> dict:
        return self._r.profile()

    @property
    def geom_count(self) -> int:
        return self._r.geom_count

    @property
    def width(self) -> int:
        return self._r.width

    @property
    def height(self) -> int:
        return self._r.height


class ParallelScene:
    """A batch of ``num_worlds`` MuJoCo worlds you step and render in one object.

    This is the high-level entry point: you only ever import ``mujofil``.
    Under the hood it drives a GPU-resident MuJoCo physics simulation and this
    package's Filament renderer for photoreal frames, handing you each
    batch of observations as a ``torch.cuda`` tensor -- no
    ``put_model`` / ``make_data`` / host-copy boilerplate in your code::

        import mujofil

        scene = mujofil.ParallelScene("scene.xml", num_worlds=32,
                                      width=256, height=256, preset="train")
        for _ in range(100):
            scene.step()                    # GPU physics
            obs = scene.render(camera=0)     # (32, 256, 256, 4) uint8 torch.cuda

    ``mujoco``, ``mujoco-warp`` and ``warp-lang`` are declared dependencies, so
    ``pip install mujofil`` already pulls them in -- there is nothing extra
    to install or import.

    To set controls / initial state, reach the GPU physics ``Data`` via
    :attr:`data` (a ``mujoco_warp`` Data) and the model via :attr:`model`.
    """

    def __init__(self, model, *, num_worlds: int = 1,
                 width: int = 256, height: int = 256,
                 preset: str | None = None, renderer: "WarpRenderer | None" = None,
                 **toggles):
        try:
            import mujoco
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "ParallelScene needs mujoco, mujoco-warp and warp-lang. These are "
                "declared dependencies of mujofil; reinstall with "
                "`pip install mujofil` to pull them in."
            ) from e
        self._mujoco, self._mjw, self._wp = mujoco, mjw, wp

        if isinstance(model, mujoco.MjModel):
            self.model = model
        elif isinstance(model, str) and "<" in model and ">" in model:
            self.model = mujoco.MjModel.from_xml_string(model)
        elif isinstance(model, str):
            self.model = mujoco.MjModel.from_xml_path(model)
        else:
            raise TypeError(
                "model must be a mujoco.MjModel, an XML file path, or an XML "
                "string; got {}".format(type(model)))

        self.num_worlds = int(num_worlds)
        self._ngeom = self.model.ngeom

        # GPU physics state (MuJoCo Warp) + a host MjData per world that the
        # renderer reads geom transforms from.
        self._M = mjw.put_model(self.model)
        self._d = mjw.make_data(self.model, nworld=self.num_worlds)
        self._host = [mujoco.MjData(self.model) for _ in range(self.num_worlds)]
        for h in self._host:
            mujoco.mj_forward(self.model, h)

        if renderer is None:
            renderer = WarpRenderer(width=width, height=height,
                                    batch_size=self.num_worlds,
                                    preset=preset, **toggles)
        elif preset or toggles:
            raise TypeError("pass either `renderer=` or render kwargs/`preset=`, not both")
        self.renderer = renderer
        self.renderer.load_model(self.model)
        self._dirty = True  # host transforms need a refresh before the next render

    # --- physics ---
    @property
    def data(self):
        """The GPU physics state (a ``mujoco_warp`` Data). Use it to set controls
        or initial state, e.g. ``scene.data.ctrl.assign(my_ctrl)``."""
        return self._d

    @property
    def warp_model(self):
        """The ``mujoco_warp`` Model put on the GPU."""
        return self._M

    def step(self, n: int = 1) -> "ParallelScene":
        """Advance the physics ``n`` steps on the GPU (MuJoCo Warp)."""
        for _ in range(n):
            self._mjw.step(self._M, self._d)
        self._dirty = True
        return self

    def reset(self) -> "ParallelScene":
        """Reset every world to the model's initial state."""
        self._d = self._mjw.make_data(self.model, nworld=self.num_worlds)
        for h in self._host:
            self._mujoco.mj_forward(self.model, h)
        self._dirty = True
        return self

    def _sync_host(self) -> None:
        # Pull the per-world geom transforms off the GPU into the host MjData the
        # renderer reads. Tiny (ngeom*12 floats/world) vs the pixels we keep on GPU.
        self._wp.synchronize()
        gx = self._d.geom_xpos.numpy()
        gm = self._d.geom_xmat.numpy().reshape(self.num_worlds, self._ngeom, 9)
        for i, h in enumerate(self._host):
            h.geom_xpos[:] = gx[i]
            h.geom_xmat[:] = gm[i]
        self._dirty = False

    # --- rendering ---
    def render(self, camera=0):
        """Render every world from ``camera`` (a camera name or id).

        Returns an ``(num_worlds, H, W, 4)`` ``uint8`` ``torch.cuda`` tensor,
        delivered zero-copy (the pixels never leave the GPU)."""
        if self._dirty:
            self._sync_host()
        return self.renderer.render_batch(self.model, self._host,
                                           cam_id=self._resolve_camera(camera))

    def _resolve_camera(self, camera) -> int:
        if isinstance(camera, str):
            cid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cid < 0:
                raise ValueError("no <camera> named {!r} in the model".format(camera))
            return cid
        return int(camera)

    # --- lifecycle ---
    def close(self) -> None:
        """Release the renderer (the GPU physics state is freed by GC)."""
        r = self.__dict__.pop("renderer", None)
        if r is not None:
            r.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _addr(obj) -> int:
    """Accept a mujoco.MjModel/MjData or a raw integer address."""
    if isinstance(obj, int):
        return obj
    a = getattr(obj, "_address", None)
    if a is not None:
        return a
    raise TypeError(f"expected mujoco struct or int address, got {type(obj)}")


def _check_cam(model, cam_id: int):
    """Fail with a clear message when a fixed camera is requested but the model
    has none (or too few). MJWarp's raycaster indexes cam arrays on the GPU, so an
    out-of-range cam_id otherwise surfaces as an opaque 'CUDA illegal access'
    much later. cam_id < 0 = free camera (set via set_free_camera), always OK.
    Raw int addresses can't be introspected, so they skip the check."""
    if cam_id < 0 or isinstance(model, int):
        return
    ncam = getattr(model, "ncam", None)
    if ncam is None:
        return
    if ncam == 0:
        raise ValueError(
            "render requested fixed camera cam_id={} but the MJCF defines no "
            "<camera>. Add a <camera> to the model, or use a free camera: pass "
            "cam_id=-1 and call set_free_camera(ex,ey,ez, tx,ty,tz).".format(cam_id))
    if cam_id >= ncam:
        raise ValueError(
            "cam_id={} is out of range; the model has {} camera(s) (valid fixed "
            "ids 0..{}). Use cam_id=-1 for a free camera.".format(
                cam_id, ncam, ncam - 1))


__all__ = ["ParallelScene", "WarpRenderer", "RendererConfig", "make_config", "QUALITY_PRESETS"]
