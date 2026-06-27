"""``mujofil-doctor``: a quick environment diagnostic for mujofil.

Checks the things mujofil needs (an NVIDIA GPU + driver, a CUDA-enabled PyTorch,
the EGL/Vulkan loaders, and that the native renderer actually initializes and
renders one frame) and prints a readable report. Run it when ``pip install
mujofil`` succeeds but rendering fails, to see which piece is missing.

    $ mujofil-doctor
"""
from __future__ import annotations

import ctypes.util
import os
import platform
import sys

_OK = "[ OK ]"
_WARN = "[WARN]"
_FAIL = "[FAIL]"


def _print(status: str, label: str, detail: str = "") -> None:
    line = f"{status}  {label}"
    if detail:
        line += f": {detail}"
    print(line)


def _check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    _print(_OK if ok else _WARN, "Python",
           f"{platform.python_version()} ({platform.system()} {platform.machine()})")
    if not ok:
        _print(_WARN, "Python version", "mujofil wheels target CPython 3.10-3.13")
    return ok


def _check_driver() -> bool:
    """An NVIDIA driver must be present (the CUDA/GL interop is NVIDIA-only)."""
    version = None
    proc = "/proc/driver/nvidia/version"
    if os.path.exists(proc):
        try:
            with open(proc) as f:
                first = f.readline().strip()
            version = first
        except OSError:
            pass
    has_lib = bool(ctypes.util.find_library("cuda") or
                   ctypes.util.find_library("nvidia-ml"))
    ok = version is not None or has_lib
    _print(_OK if ok else _FAIL, "NVIDIA driver",
           version or ("libcuda found" if has_lib else "not detected"))
    if not ok:
        _print(_FAIL, "NVIDIA driver",
               "no NVIDIA driver found; mujofil needs an NVIDIA GPU. "
               "Verify with `nvidia-smi`.")
    return ok


def _check_torch() -> bool:
    """A CUDA-enabled PyTorch is required for the zero-copy torch.cuda output."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _print(_FAIL, "PyTorch", f"not importable ({type(exc).__name__}: {exc})")
        _print(_FAIL, "PyTorch", "install a CUDA build from "
               "https://pytorch.org/get-started/locally/")
        return False
    cuda_build = getattr(torch.version, "cuda", None)
    try:
        avail = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        avail = False
    _print(_OK, "PyTorch", f"{torch.__version__} (CUDA build {cuda_build or 'none'})")
    if not avail:
        _print(_FAIL, "torch.cuda",
               "torch.cuda.is_available() is False. Common cause: the installed "
               "torch's CUDA build does not match your driver (e.g. a cu130 wheel "
               "on an older driver, or a cu124 wheel on Blackwell). Install a "
               "matching build, e.g. --index-url "
               "https://download.pytorch.org/whl/cu124 (or cu128 for Blackwell).")
        return False
    try:
        name = torch.cuda.get_device_name(0)
        cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
        _print(_OK, "CUDA device", f"{name} (sm_{cap.replace('.', '')})")
    except Exception:  # noqa: BLE001
        pass
    return True


def _check_loader(name: str, soname: str, required: bool) -> bool:
    found = ctypes.util.find_library(name)
    status = _OK if found else (_FAIL if required else _WARN)
    _print(status, f"{soname} loader", found or "not found")
    return bool(found)


def _check_render(backend: str) -> bool:
    """Construct the native renderer for ``backend`` and render one frame."""
    os.environ["MUJOFIL_BACKEND"] = backend
    try:
        import mujoco
        import mujofil
    except Exception as exc:  # noqa: BLE001
        _print(_FAIL, f"render ({backend})", f"import failed: {exc}")
        return False
    xml = ("<mujoco><worldbody>"
           "<geom type='plane' size='2 2 .1'/>"
           "<body pos='0 0 .3'><freejoint/>"
           "<geom type='box' size='.1 .1 .1' rgba='1 0 0 1'/></body>"
           "<camera name='c' pos='1.5 1.5 1' xyaxes='-1 1 0 -.4 -.4 1.6'/>"
           "</worldbody></mujoco>")
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        r = mujofil.WarpRenderer(width=64, height=64, batch_size=1)
        r.load_model(model)
        img = r.render_batch(model, [data], cam_id=0)
        ok = bool(img.is_cuda and tuple(img.shape) == (1, 64, 64, 4))
        r.close()
        _print(_OK if ok else _FAIL, f"render ({backend})",
               f"{tuple(img.shape)} {img.dtype} on {img.device}")
        return ok
    except Exception as exc:  # noqa: BLE001
        _print(_FAIL, f"render ({backend})", f"{type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print(f"mujofil-doctor\n{'=' * 40}")
    try:
        import mujofil
        _print(_OK, "mujofil", getattr(mujofil, "__version__", "?"))
    except Exception as exc:  # noqa: BLE001
        _print(_FAIL, "mujofil", f"not importable: {exc}")
        return 1

    results = [
        _check_python(),
        _check_driver(),
        _check_torch(),
    ]
    # EGL is required for the headless GL backend; Vulkan loader is optional.
    _check_loader("EGL", "libEGL.so.1", required=True)
    _check_loader("vulkan", "libvulkan.so.1", required=False)

    print(f"{'-' * 40}")
    gl_ok = _check_render("gl")
    vk_ok = _check_render("vulkan")
    print(f"{'-' * 40}")

    if gl_ok or vk_ok:
        backend = "gl" if gl_ok else "vulkan"
        print(f"{_OK}  mujofil can render on this machine (backend: {backend}).")
        return 0
    if all(results):
        print(f"{_FAIL}  Prerequisites look present but no backend rendered. "
              "See the messages above.")
    else:
        print(f"{_FAIL}  mujofil cannot render here; fix the {_FAIL} items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
