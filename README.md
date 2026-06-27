# mujofil

**A GPU simulation pipeline for vision-based RL: MuJoCo Warp physics + a parallel,
high-fidelity rasterization renderer (forked from Google Filament), zero-copy to
PyTorch.**

`mujofil` builds an **efficient, GPU-parallel rasterization render engine**
(a fork of [Google Filament](https://github.com/google/filament): PBR materials,
image-based lighting, soft shadows, SSAO, reflections) and wires it into a complete
simulation pipeline. It plugs
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)'s **high-throughput
GPU physics** into that renderer so you get the **best of both**: MuJoCo Warp's
fast, massively parallel dynamics *and* fast, parallel, photoreal visual frames
from the Filament fork, delivered **straight to PyTorch as `torch.cuda` tensors
with no CPU round-trip** for the pixels.

**What that unlocks:**

- **Drop in any environment.** Pull scenes/assets from
  [Sketchfab](https://sketchfab.com), [Poly Haven](https://polyhaven.com) and
  similar sources (glTF / GLB / OBJ / USD) and train your robot's RL policy inside
  them. These are photoreal worlds MuJoCo and MuJoCo Warp's built-in raycaster
  cannot even load.
- **Photoreal vision observations** (PBR, IBL, reflections) at parallel-batch
  throughput, so the renderer keeps up with GPU physics instead of bottlenecking it.
- **One import.** Your code only ever imports `mujofil`; it drives the
  MuJoCo Warp physics and the renderer for you.

> **Positioning, honestly:** the physics is MuJoCo Warp (DeepMind + NVIDIA's GPU
> MuJoCo); we don't reimplement dynamics. Our work is the **parallel rasterization
> renderer and the zero-copy GPU-to-PyTorch pipeline** that turns those GPU-resident
> world states into photoreal training observations. It targets the **middle of the
> fidelity/speed spectrum**: more realistic than a flat raycaster, far lighter than
> ray-traced stacks like Omniverse, photoreal-enough RGB that runs on a mid-range GPU.

> 📖 **Full documentation:** [docs/](docs/): [getting started](docs/getting-started.md),
> [API guide](docs/guide.md), [feature reference](docs/features.md),
> [cookbook & troubleshooting](docs/cookbook.md).

## Highlights

- **Zero-copy to `torch.cuda`.** Filament renders into GPU memory that CUDA
  imports directly; observations arrive as `torch.cuda` tensors with no
  GPU-to-CPU-to-GPU bounce.
- **GPU-resident pipeline.** MJWarp steps physics on the GPU; only a tiny
  transform array crosses to the host. Pixels never leave the GPU.
- **Photoreal.** Full PBR metalness/roughness, IBL, soft shadows, SSAO, MSAA,
  filmic tone mapping. Renders complete GLB environments MJWarp/MuJoCo can't.
- **Two backends.** An OpenGL single-sync path and a Vulkan shared-device path,
  selectable at runtime.

## Performance (RTX 4060 Laptop, 8 GiB)

All numbers are env-steps/s (= cameras/s), MJWarp GPU physics to `torch.cuda`.

**vs vanilla MuJoCo, same scene, same workload** (ours adds PBR + zero-copy):

| | 128px N=512 | 256px N=512 | 256px N=1024 |
|---|---|---|---|
| **mujofil (GL)** | **10,675** | **9,949** | **10,628** |
| vanilla `mujoco.Renderer` | 8,394 | 4,808 | 5,021 |
| **speedup** | **1.27×** | **2.07×** | **2.12×** |

We beat vanilla MuJoCo by **1.25 to 2.12x** on equal work; the gap widens at higher
resolution because zero-copy avoids the CPU readback that scales with pixels.

**Full photoreal warehouse** (3 GLB meshes + IBL + 16 spotlights + SSAO, geometry
vanilla MuJoCo and MJWarp cannot even load): **~3,200 cam/s** at 128px, holding flat
from N=64 to N=2048.

**GL vs Vulkan backend** (full warehouse): the GL single-sync path is **1.3×**
faster and, critically, its sync cost is **constant** across N (one `flushAndWait`),
where the Vulkan path's grows linearly with batch size.

**vs MJWarp's own raycaster:** MJWarp scales to ~42,000 cam/s at N=2048, but that
is **flat Lambertian on bare objects** (no PBR/IBL, no GLB environments). At small
N (<=32) `mujofil` is faster *and* photoreal; at large N MJWarp wins raw
throughput by trading away all visual fidelity. Different categories: MJWarp is a
parallel raycaster, this is a photoreal rasterizer.

## Quickstart

You only import `mujofil`. `ParallelScene` runs the GPU physics (MuJoCo Warp)
and renders every world to a zero-copy `torch.cuda` tensor, with no `put_model` /
`make_data` / host-copy boilerplate:

```python
import mujofil

scene = mujofil.ParallelScene("scene.xml", num_worlds=32,
                                   width=256, height=256, preset="high")

for _ in range(100):
    scene.step()                     # GPU physics (MuJoCo Warp)
    obs = scene.render(camera=0)     # (32, 256, 256, 4) uint8 torch.cuda, zero-copy
```

Set controls or initial state through `scene.data` (the MuJoCo Warp `Data`) and
the model through `scene.model`. See
[examples/minimal_render.py](examples/minimal_render.py) for a runnable demo.

<details>
<summary>Lower-level API (drive the physics yourself)</summary>

If you already run your own MuJoCo Warp loop, render a batch of host `MjData`
directly with `WarpRenderer`:

```python
import mujoco, mujoco_warp as mjw, warp as wp
from mujofil import WarpRenderer

mjm = mujoco.MjModel.from_xml_path("scene.xml")
M = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=32)
host = [mujoco.MjData(mjm) for _ in range(32)]

r = WarpRenderer(width=256, height=256, batch_size=32, preset="high")
r.load_model(mjm)

mjw.step(M, d); wp.synchronize()
gx = d.geom_xpos.numpy(); gm = d.geom_xmat.numpy().reshape(32, mjm.ngeom, 9)
for i, h in enumerate(host):
    h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]

obs = r.render_batch(mjm, host, cam_id=0)   # (32, 256, 256, 4) uint8 torch.cuda
```

</details>

## Quality toggles

Every fidelity feature is an independent toggle so you can reproduce the
throughput/fidelity trade-offs in `benchmarks/` on your own hardware:

```python
from mujofil import WarpRenderer, make_config

# keyword toggles
r = WarpRenderer(width=256, batch_size=32, ssao=False, shadows=True, msaa=True)

# or a named preset, optionally overriding individual toggles
r = WarpRenderer(width=256, batch_size=32, preset="fast")          # SSAO off, ~2x
r = WarpRenderer(width=256, batch_size=32, preset="high", bloom=True)

# or an explicit config
cfg = make_config(width=256, height=256, batch_size=32, exposure=1.6)
r = WarpRenderer(config=cfg)
```

| Toggle | Effect | Notes |
|---|---|---|
| `ssao` | screen-space ambient occlusion | **biggest cost, ~2x faster when off** |
| `ssao_quality` | SSAO quality `low`/`medium`/`high`/`ultra` | affects look more than speed |
| `ssao_ssct` | SSAO cone tracing (contact shadows) | small extra cost on top of SSAO |
| `shadows` | soft shadow maps | |
| `msaa` / `msaa_samples` | multi-sample AA | 2 / 4 / 8 |
| `bloom` | HDR bloom | off by default |
| `fxaa` | fast approximate AA | alternative to MSAA |
| `exposure` | linear exposure | before tone mapping |
| `tone_mapping` | FILMIC vs LINEAR | |
| `dithering` | temporal dithering | reduces banding |

**Presets:** `high` (photoreal, default), `medium` (high-quality SSAO, no cone
tracing), `fast` (SSAO off, ~2x), `ultra` (8x MSAA + bloom), `raw` (no AO/shadows/AA,
~3x). `eval` is an alias of `high`; `train` is an alias of `fast` tuned for vision-RL.

## Backends

Select at runtime with `MUJOFIL_BACKEND`:

- **`gl`** (default) is OpenGL single-sync, fully headless via surfaceless EGL (no
  X server needed). Renders N worlds into N imported GL textures bracketed by one
  `flushAndWait`, then exports via GL-to-CUDA interop. Sync cost is constant in N
  and it is the **fastest and most-tested path**. This is the universal default
  and fallback.
- **`vulkan`** is a shared Vulkan device + exportable swapchain + CUDA external-memory
  import. Also fully headless, but the 2-frame in-flight cap makes its sync cost
  grow with batch size. It is optional/experimental; if it cannot load or
  initialize, mujofil warns and falls back to the headless OpenGL backend.

```bash
# default is gl; force a backend explicitly with the env var:
MUJOFIL_BACKEND=gl     python examples/minimal_render.py --preset high
MUJOFIL_BACKEND=vulkan python examples/minimal_render.py --preset high
```

## Installation

```bash
pip install mujofil
```

The wheel is **self-contained**: the custom EGL-enabled Filament and the CUDA
runtime are statically baked into the native module, the compiled materials ship
inside it, and `libc++` is bundled. There is **nothing to build and no Filament,
CUDA toolkit, or graphics SDK to install separately**; the only hard requirement
at runtime is an **NVIDIA GPU + driver** and a CUDA-enabled PyTorch (pulled in
automatically, see [PyTorch](#pytorch-zero-copy-target) below).

### Supported environments

Because the package contains **no CUDA device code** (only host-side runtime
calls), a single wheel is portable across GPUs and driver versions:

| Dimension | Support |
|---|---|
| GPU | Any NVIDIA GPU (Turing / Ampere / Ada / Hopper / ...), no compute-capability lock-in |
| Driver / CUDA | NVIDIA driver **≥ R525** (CUDA 12.0+). One wheel, all newer drivers |
| OS | Linux **x86_64**, glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL/Alma/Rocky 9+, Fedora 35+) |
| Python | CPython 3.10 – 3.13 |

Not yet supported: aarch64 (Jetson/Grace), glibc < 2.34 (Ubuntu 20.04 / RHEL 8),
non-NVIDIA GPUs. These need a from-source Filament build (planned).

### PyTorch (zero-copy target)

`torch` is a dependency and is installed automatically. The default PyPI build
works for **Ada / Hopper / Ampere** GPUs. On **Blackwell** you must replace it
with a CUDA-12.8 build, because the zero-copy DLPack handoff runs CUDA kernels
through your torch, not ours:

- **Blackwell (sm_120 consumer RTX 50-series, e.g. 5090; sm_100 datacenter, e.g.
  B200):** install a **CUDA 12.8** torch, `pip install torch --index-url
  https://download.pytorch.org/whl/cu128`. A `torch+cu124` (or older) build has no
  Blackwell kernels and fails at runtime with `CUDA error: no kernel image is
  available for execution on the device`.
- **Ada / Hopper / Ampere (sm_80 to sm_90):** the default torch is fine.

`warp-lang` and `mujoco-warp` JIT-compile for the local GPU, so they need no such
pinning; only torch ships prebuilt device code. If you manage torch yourself
(common on clusters), install your CUDA-matched build first; pip will keep it.

> **Note on the default install.** On many machines `pip install mujofil` resolves
> the newest default-index torch, whose CUDA build may be newer than your driver
> (for example a `cu130` torch on an `R550` / CUDA 12.4 driver). That torch reports
> `torch.cuda.is_available() == False`; mujofil detects this at construction and
> raises a clear, actionable error (it does not crash). The fix is to install a
> torch build matching your driver, e.g. `pip install torch --index-url
> https://download.pytorch.org/whl/cu124`.

> **If the default GL backend fails to start** on an unusual GPU/driver (some
> datacenter parts report a `GL_INVALID_ENUM` at Filament `Engine::create`), the
> error now includes Filament's real message and tells you to try the headless
> Vulkan backend, which uses a different driver path: `MUJOFIL_BACKEND=vulkan`.

### Headless / display

Both backends are **fully headless**, with no X server, no display, and nothing
extra to install beyond the NVIDIA driver:

- **GL** (default) uses **surfaceless EGL**, so it renders headless at full speed
  on a bare GPU server (cloud, cluster, container). This is the recommended path
  for vision-RL training.
- **Vulkan** is also headless (shared device + exportable swapchain).

GL is the default and the universal fallback: if the optional Vulkan backend is
requested but cannot load or initialize, mujofil falls back to the headless GL
backend with a warning rather than failing.

### Building from source

Most users never need this; `pip install mujofil` ships prebuilt wheels that
already contain Filament, so **nothing below applies to a normal install**.
Build from source only to hack on the C++ or target an unsupported environment.

**Prerequisites** (the native modules and Filament are built with Clang + libc++):

| Tool | Debian/Ubuntu | RHEL/Fedora/Alma |
|---|---|---|
| Clang + libc++ dev | `clang libc++-dev libc++abi-dev` | `clang` + libc++ (LLVM release) |
| CUDA toolkit (headers + static cudart) | `nvidia-cuda-toolkit` | `cuda-cudart-devel-12-x cuda-driver-devel-12-x` |
| EGL / GL dev headers | `libegl1-mesa-dev libgl1-mesa-dev` | `mesa-libEGL-devel mesa-libGL-devel` |
| Build tools (source-built Filament only) | `git cmake ninja-build` | `git cmake ninja-build` |

Then:

```bash
git clone https://github.com/tau-intelligence/mujofil
cd mujofil
CC=clang CXX=clang++ pip install .
```

**How Filament is resolved when building from source** (the GL backend's headless
EGL rendering needs a **custom EGL-enabled Filament**, because Google's prebuilt
Linux Filament is GLX-only). This applies only to a from-source build; **prebuilt
wheels already bundle it**. `CMakeLists.txt` tries, in order:

1. **`FILAMENT_DIR=/path/to/egl-filament`** if you set it, used as-is (fastest).
2. **Download** a prebuilt EGL Filament artifact (seconds). The default path.
3. **Build from source** via `packaging/build_filament_egl.sh` (~20-30 min) if
   the download is unavailable; this is the step that needs git/cmake/ninja.

So a plain `pip install .` is **one command**; supply `FILAMENT_DIR` to skip the
download/build entirely:

```bash
CC=clang CXX=clang++ FILAMENT_DIR=/path/to/egl-filament pip install .
```

The EGL Filament artifact is reproducible from source:

```bash
packaging/build_filament_egl.sh ./_filament_egl   # clone + patch + build
```

### Dev rebuilds (no full reinstall)

For iterating on the C++ without a full `pip install`, the two helper scripts
build the modules in place (point `FILAMENT_DIR` at the EGL Filament build):

```bash
bash native/build_gl.sh   # OpenGL single-sync, headless EGL -> _mujofil_warp_gl
bash native/build.sh      # Vulkan zero-copy                  -> _mujofil_warp
```

## Architecture & porting

`mujofil` is **one core with pluggable rendering backends**, so new platforms
are added as a backend, not a fork.

```
mujofil/__init__.py     Python API, presets, backend selection   (shared)
native/render_module.cpp     pybind bindings, batching                (shared)
native/vendor/core/          scene / material / light bridge          (shared)
native/renderer_gl.cpp       Linux: surfaceless EGL  + CUDA interop   (backend)
native/renderer_warp.cpp     Linux: Vulkan device    + CUDA interop   (backend)
```

Everything platform-specific lives behind the `vf_mujoco::Renderer` interface
(context creation, GPU-to-tensor interop). Adding **macOS** or **Windows** means
adding one `renderer_*.{cpp,mm}` implementing that interface; the scene,
material, lighting, Python API, and batching layers are reused unchanged.

- **Windows** would use a WGL/EGL context + `OPAQUE_WIN32` external-memory handles
  for the CUDA interop.
- **macOS** is a different target: there is **no CUDA on Apple platforms**, so a
  Mac backend would use Filament's **Metal** backend and export to PyTorch via
  **MPS** (`MTLBuffer` to torch-MPS) rather than `torch.cuda`.

These are not yet implemented (they need the respective hardware to develop and
validate on), but the codebase is structured so they slot in without a fork.

## Layout

```
mujofil/        Python package (WarpRenderer, make_config, presets)
native/              C++ renderer + pybind module + build scripts
  renderer_gl.cpp      OpenGL single-sync zero-copy backend
  renderer_warp.cpp    Vulkan shared-device zero-copy backend
  render_module.cpp    pybind bindings (shared by both backends)
examples/            runnable demos
benchmarks/          the benchmark suite behind the numbers above
spikes/              isolated feasibility proofs (GL/CUDA, Vulkan/CUDA, DLPack)
docs/ARCHITECTURE.md design + phased integration plan
```

## Provenance

`mujofil` is **this** package: GPU-resident MuJoCo Warp physics + the parallel
Filament-fork rasterizer + zero-copy `torch.cuda` output. Its renderer reuses the
scene/material/light bridge originally written for the CPU-MuJoCo renderer, but
builds it into a separate GPU pipeline. The earlier CPU-physics edition (NumPy
frames, `mujocofil` on the CPU) has been retired and folded into this package, so
there is now a single `mujofil` to install.

## License

Apache-2.0.
