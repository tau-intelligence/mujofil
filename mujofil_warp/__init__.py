"""mujofil-warp: photoreal PBR rendering for GPU-resident MuJoCo (MJWarp).

Renders MuJoCo scenes with Google Filament (PBR + IBL) on a Vulkan device shared
with CUDA, so rendered frames are delivered straight to PyTorch as CUDA tensors
with NO CPU round-trip (zero-copy).

This package uses mujofil's renderer source (SceneBridge/materials/lights) but is
a SEPARATE build — mujofil's PyPI release is untouched.
"""
from __future__ import annotations

import os
import sys

__version__ = "0.1.2"

# Locate the native modules. When installed (pip), the compiled
# _mujofil_warp_gl / _mujofil_warp .so live INSIDE this package directory. In a
# dev checkout they're built into ../native by native/build*.sh — support both.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
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
    """Select the native backend: 'gl' (default, OpenGL single-sync) or 'vulkan'.

    The GL backend (`_mujofil_warp_gl`) renders N worlds into N GL textures with a
    single flushAndWait (true single-sync) and exports them to CUDA via GL interop.
    It is the fastest path and is fully **headless** (surfaceless EGL — no X server
    required). The Vulkan backend (`_mujofil_warp`) uses a shared device +
    exportable swapchain and is also headless. Override with
    MUJOFIL_WARP_BACKEND=gl|vulkan.

    Default is 'gl'. If 'gl' is requested implicitly (no explicit override) but the
    GL module isn't built / can't initialize, we fall back to Vulkan.
    """
    explicit = backend is not None or "MUJOFIL_WARP_BACKEND" in os.environ
    backend = (backend or os.environ.get("MUJOFIL_WARP_BACKEND", "gl")).lower()

    if backend in ("vulkan", "vk"):
        import _mujofil_warp as _vk  # noqa: E402
        return _vk

    # GL path (default), headless via EGL. Fall back to Vulkan only when GL wasn't
    # explicitly asked for and its module is missing/unimportable.
    if backend in ("gl", "opengl"):
        try:
            import _mujofil_warp_gl as _gl  # noqa: E402
            return _gl
        except Exception:
            if explicit:
                raise
            import _mujofil_warp as _vk  # noqa: E402
            return _vk

    raise ValueError(f"unknown MUJOFIL_WARP_BACKEND={backend!r} (use 'gl' or 'vulkan')")


_native = _load_native()
RendererConfig = _native.RendererConfig


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
    # --- tone mapping / exposure ---
    exposure: float = 0.0,
    tone_mapping: bool = True,
    dithering: bool = True,
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
    cfg = RendererConfig()
    cfg.width = width
    cfg.height = height
    cfg.batch_size = batch_size
    cfg.enable_ssao = ssao
    cfg.ssao_quality = ssao_q
    cfg.ssao_ssct = ssao_ssct
    cfg.enable_shadows = shadows
    cfg.enable_msaa = msaa
    cfg.msaa_samples = msaa_samples
    cfg.enable_bloom = bloom
    cfg.enable_fxaa = fxaa
    cfg.exposure = exposure
    cfg.tone_mapping = tone_mapping
    cfg.dithering = dithering
    return cfg


# Named quality presets so users can reproduce our benchmark trends on their own
# hardware. Pass to ``WarpRenderer(preset=...)``; override individual toggles too.
#   high    -- full photoreal PBR (ULTRA SSAO + cone tracing). The default.
#   medium  -- HIGH-quality SSAO, no cone tracing: nearly identical look, a little
#              faster (the SSAO *pass* dominates, not its quality level).
#   fast    -- SSAO off (~2x throughput), shadows + MSAA kept. The big lever.
#   ultra   -- high + bloom + 8x MSAA.
#   raw     -- no AO / shadows / AA (maximum throughput, ~3x).
QUALITY_PRESETS = {
    "high": dict(ssao=True, ssao_quality="ultra", ssao_ssct=True,
                 shadows=True, msaa=True, msaa_samples=4),
    "medium": dict(ssao=True, ssao_quality="high", ssao_ssct=False,
                   shadows=True, msaa=True, msaa_samples=4),
    "fast": dict(ssao=False, shadows=True, msaa=True, msaa_samples=4),
    "ultra": dict(ssao=True, ssao_quality="ultra", ssao_ssct=True,
                  shadows=True, msaa=True, msaa_samples=8, bloom=True),
    "raw": dict(ssao=False, shadows=False, msaa=False),
}


class WarpRenderer:
    """PBR renderer that returns frames as torch CUDA tensors (zero-copy).

    Construct it three ways::

        # 1. keyword toggles (recommended)
        r = WarpRenderer(width=256, height=256, batch_size=32, ssao=False)

        # 2. a named quality preset ("high" | "fast" | "ultra" | "raw")
        r = WarpRenderer(width=256, batch_size=32, preset="fast")

        # 3. an explicit RendererConfig
        r = WarpRenderer(config=make_config(width=256, ssao=True))

    Then::

        r.load_model(model)                # mujoco.MjModel
        r.sync_camera(model, data, cam_id=0)
        img = r.render()                   # (H, W, 4) uint8 torch.cuda tensor

    Quality toggles (``ssao``, ``shadows``, ``msaa``, ``bloom``, ``fxaa``,
    ``exposure``, ``tone_mapping``, ``dithering``) are documented on
    :func:`make_config`. ``ssao`` is the biggest throughput lever (~2x off).
    """

    def __init__(self, config: "RendererConfig | None" = None,
                 *, preset: str | None = None, **toggles):
        if config is None:
            kw = dict(QUALITY_PRESETS[preset]) if preset else {}
            kw.update(toggles)
            config = make_config(**kw)
        elif preset or toggles:
            raise TypeError("pass either `config=` or keyword toggles/`preset=`, not both")
        self._r = _native.WarpRenderer(config)

    # --- scene ---
    def load_model(self, model):
        self._r.load_model(_addr(model))

    def load_glb(self, path: str):
        self._r.load_glb(path)

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
        self._r.sync_camera(_addr(model), _addr(data), cam_id)

    def render(self):
        """Render and return an (H, W, 4) uint8 torch.cuda tensor (zero-copy)."""
        import torch
        return torch.from_dlpack(self._r.render_dlpack())

    def render_batch(self, model, datas, cam_id: int = -1):
        """Render N worlds (one MjData each) with ONE GPU sync.

        Returns an (N, H, W, 4) uint8 torch.cuda tensor (zero-copy). The renderer
        must have been created with config.batch_size >= len(datas).
        """
        import torch
        ptrs = [_addr(d) for d in datas]
        return torch.from_dlpack(self._r.render_batch_dlpack(_addr(model), ptrs, cam_id))

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


def _addr(obj) -> int:
    """Accept a mujoco.MjModel/MjData or a raw integer address."""
    if isinstance(obj, int):
        return obj
    a = getattr(obj, "_address", None)
    if a is not None:
        return a
    raise TypeError(f"expected mujoco struct or int address, got {type(obj)}")


__all__ = ["WarpRenderer", "RendererConfig", "make_config", "QUALITY_PRESETS"]
