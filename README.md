# mujofil-warp

**Photoreal PBR rendering for GPU-resident MuJoCo (MJWarp), zero-copy to PyTorch.**

[MJWarp](https://github.com/google-deepmind/mujoco_warp) simulates thousands of
parallel MuJoCo worlds entirely on the GPU, but its built-in batch renderer is a
deliberately **low-fidelity single-hit raycaster** (flat Lambertian, no PBR / IBL
/ reflections, and it cannot load GLB environments).

`mujofil-warp` pairs MJWarp's GPU-resident physics with
[Google Filament](https://github.com/google/filament)'s **physically-based
renderer** (PBR materials, image-based lighting, soft shadows, SSAO) and delivers
each rendered frame **straight to PyTorch as a CUDA tensor — no CPU round-trip**.

## Highlights

- **Zero-copy to `torch.cuda`.** Filament renders into GPU memory that CUDA
  imports directly; observations arrive as `torch.cuda` tensors with no
  GPU→CPU→GPU bounce.
- **GPU-resident pipeline.** MJWarp steps physics on the GPU; only a tiny
  transform array crosses to the host. Pixels never leave the GPU.
- **Photoreal.** Full PBR metalness/roughness, IBL, soft shadows, SSAO, MSAA,
  filmic tone mapping — renders complete GLB environments MJWarp/MuJoCo can't.
- **Two backends.** An OpenGL single-sync path and a Vulkan shared-device path,
  selectable at runtime.

## Performance (RTX 4060 Laptop, 8 GiB)

All numbers are env-steps/s (= cameras/s), MJWarp GPU physics → `torch.cuda`.

**vs vanilla MuJoCo, same scene, same workload** (ours adds PBR + zero-copy):

| | 128px N=512 | 256px N=512 | 256px N=1024 |
|---|---|---|---|
| **mujofil-warp (GL)** | **10,675** | **9,949** | **10,628** |
| vanilla `mujoco.Renderer` | 8,394 | 4,808 | 5,021 |
| **speedup** | **1.27×** | **2.07×** | **2.12×** |

We beat vanilla MuJoCo by **1.25–2.12×** on equal work — the gap widens at higher
resolution because zero-copy avoids the CPU readback that scales with pixels.

**Full photoreal warehouse** (3 GLB meshes + IBL + 16 spotlights + SSAO — geometry
vanilla MuJoCo and MJWarp cannot even load): **~3,200 cam/s** at 128px, holding flat
from N=64 to N=2048.

**GL vs Vulkan backend** (full warehouse): the GL single-sync path is **1.3×**
faster and, critically, its sync cost is **constant** across N (one `flushAndWait`),
where the Vulkan path's grows linearly with batch size.

**vs MJWarp's own raycaster:** MJWarp scales to ~42,000 cam/s at N=2048 — but that
is **flat Lambertian on bare objects** (no PBR/IBL, no GLB environments). At small
N (≤32) `mujofil-warp` is faster *and* photoreal; at large N MJWarp wins raw
throughput by trading away all visual fidelity. Different categories: MJWarp is a
parallel raycaster, this is a photoreal rasterizer.

## Quickstart

```python
import mujoco, mujoco_warp as mjw, warp as wp, torch
from mujofil_warp import WarpRenderer

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

See [examples/minimal_render.py](examples/minimal_render.py) for a runnable demo.

## Quality toggles

Every fidelity feature is an independent toggle so you can reproduce the
throughput/fidelity trade-offs in `benchmarks/` on your own hardware:

```python
from mujofil_warp import WarpRenderer, make_config

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
| `ssao` | screen-space ambient occlusion | **biggest cost — ~2× faster when off** |
| `shadows` | soft shadow maps | |
| `msaa` / `msaa_samples` | multi-sample AA | 2 / 4 / 8 |
| `bloom` | HDR bloom | off by default |
| `fxaa` | fast approximate AA | alternative to MSAA |
| `exposure` | linear exposure | before tone mapping |
| `tone_mapping` | FILMIC vs LINEAR | |
| `dithering` | temporal dithering | reduces banding |

**Presets:** `high` (photoreal), `fast` (SSAO off, ~2×), `ultra` (8× MSAA + bloom),
`raw` (no AO/shadows/AA).

## Backends

Select at runtime with `MUJOFIL_WARP_BACKEND`:

- **`gl`** — OpenGL single-sync. Renders N worlds into N imported GL textures
  bracketed by one `flushAndWait`, then exports via GL↔CUDA interop. Sync cost is
  constant in N; **fastest in the warehouse.** Requires an X display (`DISPLAY`).
- **`vulkan`** (default) — shared Vulkan device + exportable swapchain + CUDA
  external-memory import. Works headless (no X), but the 2-frame in-flight cap
  makes its sync cost grow with batch size.

```bash
MUJOFIL_WARP_BACKEND=gl python examples/minimal_render.py --preset high
```

## Building the native module

The native extension links a prebuilt [Filament](https://github.com/google/filament)
(1.56.x) and reuses the `mujofil` renderer source. Set `FILAMENT_DIR` to a prebuilt
Filament release, then:

```bash
bash native/build.sh      # Vulkan zero-copy   -> _mujofil_warp
bash native/build_gl.sh   # OpenGL single-sync -> _mujofil_warp_gl
```

Requirements: an NVIDIA GPU with CUDA, `clang++`/libc++, the CUDA toolkit headers,
and (for the GL backend) GLX + X11.

## Layout

```
mujofil_warp/        Python package (WarpRenderer, make_config, presets)
native/              C++ renderer + pybind module + build scripts
  renderer_gl.cpp      OpenGL single-sync zero-copy backend
  renderer_warp.cpp    Vulkan shared-device zero-copy backend
  render_module.cpp    pybind bindings (shared by both backends)
examples/            runnable demos
benchmarks/          the benchmark suite behind the numbers above
spikes/              isolated feasibility proofs (GL↔CUDA, Vulkan↔CUDA, DLPack)
docs/ARCHITECTURE.md design + phased integration plan
```

## Relationship to `mujofil`

`mujofil-warp` reuses the CPU-MuJoCo `mujofil` renderer's scene/material/light
source but is a **separate build** — the published `mujofil` package is untouched.
Use `mujofil` for high-fidelity CPU-MuJoCo vector-env rendering; use
`mujofil-warp` when you want MJWarp's GPU-resident physics with photoreal,
zero-copy observations.

## License

Apache-2.0.
