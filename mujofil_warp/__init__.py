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

# Locate the native module (built in ../native by native/build.sh).
_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE = os.path.normpath(os.path.join(_HERE, os.pardir, "native"))
if _NATIVE not in sys.path:
    sys.path.insert(0, _NATIVE)

# mujofil's MaterialManager loads compiled .filamat via VF_MUJOCO_MATERIALS_DIR.
if "VF_MUJOCO_MATERIALS_DIR" not in os.environ:
    try:
        import mujofil as _mujofil
        os.environ["VF_MUJOCO_MATERIALS_DIR"] = os.path.join(
            os.path.dirname(_mujofil.__file__), "materials")
    except Exception:  # noqa: BLE001
        pass

import _mujofil_warp as _native  # noqa: E402

RendererConfig = _native.RendererConfig


class WarpRenderer:
    """PBR renderer that returns frames as torch CUDA tensors (zero-copy).

    Example
    -------
        cfg = RendererConfig(); cfg.width = cfg.height = 256
        r = WarpRenderer(cfg)
        r.load_model(model.address)        # mujoco.MjModel
        r.sync_transforms(model, data)
        r.sync_camera(model, data, cam_id=0)
        img = r.render()                   # (H, W, 4) uint8 torch.cuda tensor
    """

    def __init__(self, config: RendererConfig | None = None):
        self._r = _native.WarpRenderer(config or RendererConfig())

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


__all__ = ["WarpRenderer", "RendererConfig"]
