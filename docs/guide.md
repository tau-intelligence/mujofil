# `mujofil-warp` — GPU MuJoCo (MJWarp) + Filament PBR, zero-copy to PyTorch

The full API. [MJWarp](https://github.com/google-deepmind/mujoco_warp) simulates
thousands of MuJoCo worlds entirely on the GPU; `mujofil-warp` renders them with
Filament PBR and delivers each frame **straight to PyTorch as a `torch.cuda`
tensor — no GPU→CPU→GPU round-trip**.

> **CPU physics?** For normal `mujoco.mj_step` rendering to NumPy frames, see the
> CPU edition [`mujofil`](https://github.com/tau-intelligence/MuJoCo-Filament)
> and its [guide](https://github.com/tau-intelligence/MuJoCo-Filament/blob/main/docs/guide.md).

**Contents**

- [Install](#install)
- [The core loop](#the-core-loop)
- [`WarpRenderer` API](#warprenderer-api)
- [Presets: `train` vs `eval`](#presets-train-vs-eval)
- [Quality toggles](#quality-toggles)
- [Batched zero-copy rendering](#batched-zero-copy-rendering)
- [Lights, IBL & GLB environments](#lights-ibl--glb-environments)
- [Headless & backends](#headless--backends)
- [Profiling](#profiling)
- [Performance](#performance)

---

## Install

```bash
pip install mujofil-warp
```

Runtime requirement: **an NVIDIA GPU + driver ≥ R525 (CUDA 12.0+)**. No CUDA
toolkit, no Filament, no `mujofil`. You also need `mujoco_warp` and a CUDA build
of `torch`.

```python
from mujofil_warp import WarpRenderer, make_config, QUALITY_PRESETS
```

---

## The core loop

Step physics on GPU → copy the small transform arrays to host `MjData` → render →
get a `torch.cuda` tensor. Only `ngeom × 12` floats cross to the host per world;
the *pixels never leave the GPU*.

```python
import mujoco, mujoco_warp as mjw, warp as wp, torch
from mujofil_warp import WarpRenderer

N = 32
mjm = mujoco.MjModel.from_xml_path("scene.xml")   # MUST contain a <camera>
M   = mjw.put_model(mjm)
d   = mjw.make_data(mjm, nworld=N)
host = [mujoco.MjData(mjm) for _ in range(N)]

r = WarpRenderer(width=256, height=256, batch_size=N, preset="eval")
r.load_model(mjm)

for step in range(1000):
    mjw.step(M, d); wp.synchronize()

    gx = d.geom_xpos.numpy()
    gm = d.geom_xmat.numpy().reshape(N, mjm.ngeom, 9)
    for i, h in enumerate(host):
        h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]

    obs = r.render_batch(mjm, host, cam_id=0)   # (N,256,256,4) uint8 torch.cuda
```

> **Always include a `<camera>` in the MJCF** — the renderer indexes camera 0; a
> scene with none reads invalid memory → CUDA illegal-access.

---

## `WarpRenderer` API

Construct it three ways:

```python
# 1. keyword toggles (recommended)
r = WarpRenderer(width=256, height=256, batch_size=32, ssao=False)

# 2. a named preset (override individual toggles too)
r = WarpRenderer(width=256, batch_size=32, preset="train")
r = WarpRenderer(width=256, batch_size=32, preset="eval", bloom=True)

# 3. an explicit config
from mujofil_warp import make_config
r = WarpRenderer(config=make_config(width=256, batch_size=32, exposure=1.6))
```

### Methods

| Method | Returns | Description |
|---|---|---|
| `load_model(model)` | — | Load a `mujoco.MjModel` (or its `_address`). |
| `load_glb(path)` | — | Load a GLB environment backdrop. |
| `load_ibl(ibl_ktx, skybox_ktx)` | — | Load an HDR environment (IBL + reflections). |
| `set_ambient_intensity(intensity)` | — | Scale the indirect (IBL) light. |
| `clear_dynamic_lights()` | — | Remove the auto lights (keep IBL). |
| `add_directional_light(dx,dy,dz, r,g,b, intensity, cast_shadows=False)` | — | Sun / fill (lux). |
| `add_point_light(x,y,z, r,g,b, intensity, falloff)` | — | Point light. |
| `add_spot_light(x,y,z, dx,dy,dz, r,g,b, intensity, falloff, inner_deg, outer_deg, focused=True)` | — | Spot (cone ≤ 90°). |
| `set_free_camera(ex,ey,ez, tx,ty,tz)` | — | Free camera eye + look-at. |
| `sync_transforms(model, data)` | — | Push one world's geom poses (single-frame path). |
| `sync_camera(model, data, cam_id=-1)` | — | Drive the camera from a MuJoCo camera. |
| `render()` | `(H,W,4)` torch.cuda | Render the synced single world (zero-copy). |
| `render_batch(model, datas, cam_id=-1)` | `(N,H,W,4)` torch.cuda | Render N worlds with **one GPU sync** (zero-copy). `N ≤ batch_size`. |
| `reset_profile()` / `profile()` | — / dict | Per-stage GPU timers (see [Profiling](#profiling)). |
| `geom_count`, `width`, `height` | int | Properties. |

The returned tensor is **RGBA `(…,4)` uint8**, already on `cuda:0`. Convert to
the channel order your model wants, e.g.
`obs[..., :3].permute(0,3,1,2).float()/255`.

---

## Presets: `train` vs `eval`

Two presets cover the two real use-cases. **Use these.**

| Preset | Fidelity | Use for |
|---|---|---|
| **`eval`** | Full photoreal — ULTRA SSAO + cone tracing + 4× MSAA + shadows + PBR + IBL | Evaluation videos, cinematics, sim-to-real demos — anything a **human** watches. |
| **`train`** | SSAO **off**, MSAA **off** (keeps PBR materials, IBL, shadows) | Batched vision-RL observations. **~2× faster** than `eval` (RTX 4060, 128px N=256: 8196 vs 4008 cam/s) with no impact on what a CNN learns. |

```python
train_r = WarpRenderer(width=128, batch_size=256, preset="train")  # RL rollout
eval_r  = WarpRenderer(width=512, batch_size=8,   preset="eval")   # eval video
```

> `train` spends fidelity only where a policy perceives it — SSAO subtlety and 4×
> MSAA edges are imperceptible at 128px. If you also *evaluate on pixels for
> reward*, keep the same preset for that path so the observation distribution
> matches.

### All presets

| Preset | SSAO | Shadows | MSAA | Extra | Relative speed |
|---|---|---|---|---|---|
| `eval` / `high` | ultra + cone | ✓ | 4× | — | 1× (baseline fidelity) |
| `medium` | high, no cone | ✓ | 4× | — | slightly faster |
| `fast` | off | ✓ | 4× | — | ~2× |
| `train` | off | ✓ | off | — | ~2× |
| `ultra` | ultra + cone | ✓ | 8× | bloom | slower (showcase) |
| `raw` | off | off | off | — | ~3× (fastest) |

---

## Quality toggles

Every fidelity feature is an independent keyword (also usable via `make_config`):

```python
from mujofil_warp import WarpRenderer, make_config

r = WarpRenderer(width=256, batch_size=32,
                 ssao=True, ssao_quality="ultra", ssao_ssct=True,
                 shadows=True, msaa=True, msaa_samples=4,
                 bloom=False, fxaa=False,
                 exposure=0.0, tone_mapping=True, dithering=True)
```

| Toggle | Default | Effect |
|---|---|---|
| `ssao` | `True` | Screen-space ambient occlusion. **Biggest cost — ~2× faster off.** |
| `ssao_quality` | `"ultra"` | `low`/`medium`/`high`/`ultra` (or 0–3). Affects look more than speed. |
| `ssao_ssct` | `True` | SSAO cone tracing (extra contact shadows). Small cost on top of SSAO. |
| `shadows` | `True` | Soft shadow maps. |
| `msaa` / `msaa_samples` | `True` / 4 | Multisample AA (2/4/8). |
| `bloom` | `False` | HDR bloom. |
| `fxaa` | `False` | Fast approximate AA (alternative to MSAA). |
| `exposure` | `0.0` | Linear exposure before tone mapping. |
| `tone_mapping` | `True` | FILMIC vs LINEAR. |
| `dithering` | `True` | Temporal dithering (reduces banding). |

`make_config` also takes `width`, `height`, and `batch_size`.

---

## Batched zero-copy rendering

`render_batch` renders N worlds and synchronizes **once** for the whole batch,
returning `(N, H, W, 4)` on the GPU.

```python
r = WarpRenderer(width=128, height=128, batch_size=512, preset="train")
r.load_model(mjm)
# ... step MJWarp, fill host[0..N] ...
obs = r.render_batch(mjm, host, cam_id=0)   # (N,128,128,4) torch.cuda
```

- `batch_size` (set at construction) is the **max** N. Pass `N ≤ batch_size`
  `MjData` to `render_batch`.
- The frame is true zero-copy: Filament renders into GPU memory that CUDA imports
  directly and wraps as a torch tensor via DLPack.

---

## Lights, IBL & GLB environments

Same lighting/IBL/GLB primitives as the CPU edition. The headline use is dropping
MJWarp-stepped objects into a **photoreal GLB environment** that MuJoCo/MJWarp
cannot render:

```python
r = WarpRenderer(width=256, batch_size=32, preset="eval")
r.load_model(mjm)                      # the MJWarp objects
r.load_ibl("warehouse_ibl.ktx", "warehouse_skybox.ktx")
r.set_ambient_intensity(6500.0)
r.load_glb("warehouse.glb")            # the environment backdrop
r.clear_dynamic_lights()
r.add_directional_light(-0.4,-0.5,-0.75, 1,0.97,0.92, 70000.0, True)
```

This is where PBR clearly wins: chrome/gold/metal objects mirror-reflect the
warehouse via IBL — invisible on the flat raycaster.

---

## Headless & backends

Both backends run **fully headless** (no X server). Select with
`MUJOFIL_WARP_BACKEND`:

| Backend | Default | How | Notes |
|---|---|---|---|
| **`gl`** | ✓ | Surfaceless EGL, renders N worlds into N GL textures bracketed by **one** `flushAndWait`, exports via GL↔CUDA interop. | **Fastest** (single-sync; sync cost constant in N). Headless via surfaceless EGL. |
| **`vulkan`** | | Shared Vulkan device + exportable swapchain + CUDA external-memory import. | Also fully headless. The 2-frame in-flight cap makes its sync cost grow with N. |

```bash
MUJOFIL_WARP_BACKEND=gl     python my_train.py     # default, fastest
MUJOFIL_WARP_BACKEND=vulkan python my_train.py     # force Vulkan
```

`gl` auto-falls back to Vulkan only if the GL module can't initialize. For cloud
/ cluster / container GPU servers, the default `gl` (surfaceless EGL) is the
recommended path.

---

## Profiling

Per-stage GPU timers expose where time goes (render vs flush vs copy):

```python
r.reset_profile()
for _ in range(50):
    obs = r.render_batch(mjm, host, cam_id=0)
print(r.profile())   # {'t_render':..., 't_flush':..., 't_copy':...} ms
```

`t_copy` (the zero-copy image→buffer copy) is typically negligible (~0.25 ms);
the flush/sync stage dominates at high N — which is why the GL single-sync
backend is the default.

---

## Performance

RTX 4060 Laptop, env-steps/s (= cameras/s), MJWarp GPU physics → `torch.cuda`.

**vs vanilla MuJoCo, same scene, same workload** (ours adds PBR + zero-copy):

| | 128px N=512 | 256px N=512 | 256px N=1024 |
|---|---|---|---|
| **mujofil-warp (GL)** | **10,675** | **9,949** | **10,628** |
| vanilla `mujoco.Renderer` | 8,394 | 4,808 | 5,021 |
| **speedup** | **1.27×** | **2.07×** | **2.12×** |

The gap widens with resolution because zero-copy avoids the CPU readback that
scales with pixels.

**Full photoreal warehouse** (3 GLB meshes + IBL + 16 spotlights + SSAO —
geometry vanilla MuJoCo/MJWarp can't even load): **~3,200 cam/s** at 128px, flat
from N=64 to N=2048.

**vs MJWarp's own raycaster:** MJWarp scales to ~42,000 cam/s at N=2048 — but
that's **flat Lambertian on bare objects**. At N≤32 `mujofil-warp` is faster *and*
photoreal; at very large N MJWarp wins raw throughput by trading away all visual
fidelity. Different categories: MJWarp is a parallel raycaster, this is a
photoreal rasterizer.

**Honest ceiling:** the GL path plateaus around 3,000–3,200 cam/s on the
warehouse because Filament rasterizes the N cameras sequentially. It's a unique
capability — GPU-resident physics + photoreal PBR + zero-copy — not a
raw-throughput crown.

For what each feature costs visually, see [features.md](features.md). For the
design and zero-copy internals, see [ARCHITECTURE.md](ARCHITECTURE.md).
