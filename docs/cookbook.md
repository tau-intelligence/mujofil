# Cookbook & Troubleshooting

Copy-pasteable recipes for common tasks, then a troubleshooting table.

> Doing normal CPU MuJoCo physics? See the CPU edition's cookbook:
> [`mujofil` cookbook](https://github.com/tau-intelligence/MuJoCo-Filament/blob/main/docs/cookbook.md).

**Recipes**

- [Photoreal vision-RL observations](#photoreal-vision-rl-observations)
- [`train` vs `eval` in one training run](#train-vs-eval-in-one-training-run)
- [Drop objects into a photoreal warehouse](#drop-objects-into-a-photoreal-warehouse)
- [Save an eval frame to PNG](#save-an-eval-frame-to-png)
- [Pick a backend](#pick-a-backend)
- [Profile the render](#profile-the-render)

**[Troubleshooting](#troubleshooting)**

---

## Photoreal vision-RL observations

```python
import mujoco, mujoco_warp as mjw, warp as wp
from mujofil_warp import WarpRenderer

N = 256
mjm = mujoco.MjModel.from_xml_path("scene.xml")   # needs a <camera>
M, d = mjw.put_model(mjm), mjw.make_data(mjm, nworld=N)
host = [mujoco.MjData(mjm) for _ in range(N)]

r = WarpRenderer(width=128, height=128, batch_size=N, preset="train")
r.load_model(mjm)

def rollout_step():
    mjw.step(M, d); wp.synchronize()
    gx = d.geom_xpos.numpy(); gm = d.geom_xmat.numpy().reshape(N, mjm.ngeom, 9)
    for i, h in enumerate(host):
        h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]
    return r.render_batch(mjm, host, cam_id=0)     # (N,128,128,4) torch.cuda
```

The returned tensor is already on `cuda:0`. For a CNN expecting `NCHW` floats:

```python
obs = rollout_step()[..., :3].permute(0, 3, 1, 2).float().div_(255)
```

---

## `train` vs `eval` in one training run

Fast observations for learning, full fidelity for the eval video — **same scene,
two renderers**:

```python
from mujofil_warp import WarpRenderer

train_r = WarpRenderer(width=128, height=128, batch_size=256, preset="train")
eval_r  = WarpRenderer(width=512, height=512, batch_size=4,   preset="eval")
train_r.load_model(mjm); eval_r.load_model(mjm)

obs   = train_r.render_batch(mjm, host[:256], cam_id=0)   # cheap, ~2× faster
frame = eval_r.render_batch(mjm, host[:4],   cam_id=0)    # photoreal for video
```

`train` keeps PBR/IBL/shadows but drops SSAO + MSAA (imperceptible at 128px to a
CNN, ~2× faster). `eval` is the full photoreal look.

---

## Drop objects into a photoreal warehouse

The headline use: MJWarp-stepped objects inside a GLB environment MuJoCo/MJWarp
cannot render. Metals reflect the warehouse via IBL.

```python
r = WarpRenderer(width=256, batch_size=32, preset="eval")
r.load_model(mjm)                              # the MJWarp objects
r.load_ibl("warehouse_ibl.ktx", "warehouse_skybox.ktx")
r.set_ambient_intensity(6500.0)
r.load_glb("warehouse.glb")                    # the environment backdrop
r.clear_dynamic_lights()
r.add_directional_light(-0.4,-0.5,-0.75, 1,0.97,0.92, 70000.0, True)

obs = r.render_batch(mjm, host, cam_id=0)
```

---

## Save an eval frame to PNG

```python
from PIL import Image

obs = r.render_batch(mjm, host[:1], cam_id=0)   # (1,H,W,4) torch.cuda
img = obs[0, ..., :3].cpu().numpy()             # to host only for saving
Image.fromarray(img).save("eval_frame.png")
```

(Only the *save* path copies to host; training never does.)

---

## Pick a backend

```bash
# default GL (surfaceless EGL, single-sync, fastest, headless):
python my_train.py
# force Vulkan (also headless; useful if GL can't init):
MUJOFIL_WARP_BACKEND=vulkan python my_train.py
```

GL auto-falls back to Vulkan if its module can't initialize. On cloud / cluster /
container GPU servers, the default GL works headless — nothing extra to install
beyond the NVIDIA driver.

---

## Profile the render

```python
r.reset_profile()
for _ in range(50):
    r.render_batch(mjm, host, cam_id=0)
print(r.profile())   # {'t_render':..., 't_flush':..., 't_copy':...} ms
```

`t_copy` (zero-copy) is tiny; `t_flush` dominates at high N — the GL single-sync
backend keeps it constant.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **CUDA illegal access** during render | MJCF has no `<camera>` | Add a `<camera>` (the renderer indexes camera 0). |
| `import` fails / no CUDA | No NVIDIA driver, or non-CUDA `torch` | Install driver ≥ R525; install a CUDA build of `torch`. |
| Falls back to Vulkan unexpectedly | GL module couldn't init (e.g. no EGL device) | Fine if headless; force `MUJOFIL_WARP_BACKEND=gl` to see the error. |
| Slow at large N | Vulkan backend's linear sync cost | Use the default `gl` backend (single-sync). |
| Identical frames across batch worlds | Flushing too few in-flight frames | Don't set `MUJOFIL_WARP_FLUSH_EVERY` > 2 (Vulkan 2-frame in-flight cap). |
| Metals render **black** | Nothing to reflect | `load_ibl(...)` or `load_glb(...)` so metals mirror an environment. |
| Materials look **flat / washed-out** | No IBL loaded | `load_ibl(...)` before `set_ambient_intensity`. |
| Image too **bright / dark** | Exposure | Adjust `exposure` (default 0.0). |
| **Shadows invisible** | Ambient/IBL too high | Lower ambient; use an oblique key with `cast_shadows=True`. |
| Throughput lower than expected | SSAO on | Use `preset="train"` (SSAO off, ~2×) for observations. |
| `render_batch` raises a size error | `len(datas)` > `batch_size` | Construct with `batch_size ≥ N`. |

For what each feature costs, see [features.md](features.md). For the zero-copy
internals, see [ARCHITECTURE.md](ARCHITECTURE.md).
