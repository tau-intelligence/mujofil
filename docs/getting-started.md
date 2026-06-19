# Getting Started

From nothing to a zero-copy `torch.cuda` render.

> ### Running normal CPU MuJoCo?
> This page covers **`mujofil-warp`** (GPU MJWarp physics → `torch.cuda` frames).
> If you step physics with `mujoco.mj_step` on the CPU and want **NumPy** frames,
> use the CPU edition,
> **[`mujofil`](https://github.com/tau-intelligence/MuJoCo-Filament)**, instead.

---

## 1. Is this the right package?

Use **`mujofil-warp`** when you simulate **thousands of worlds on the GPU** with
[MJWarp](https://github.com/google-deepmind/mujoco_warp) (`mujoco_warp`) and want
observations delivered straight to PyTorch as CUDA tensors with no CPU copy.

If your physics runs on the CPU (`mujoco.mj_step`), use the CPU edition,
[`mujofil`](https://github.com/tau-intelligence/MuJoCo-Filament), which returns
NumPy frames. You can install both; they don't conflict.

---

## 2. Install

```bash
pip install mujofil-warp
```

The wheel is **self-contained**: Filament and the CUDA runtime are statically
baked in, and the compiled materials ship inside it. There is **no CUDA toolkit,
no Filament, and no `mujofil`** to install. You also need `mujoco_warp` and a
CUDA build of `torch` in your environment.

---

## 3. The one requirement pip can't satisfy: an NVIDIA driver

The only hard runtime requirement is an **NVIDIA GPU + driver ≥ R525 (CUDA
12.0+)**. Because the package contains **no CUDA device code**, one wheel runs on
any NVIDIA GPU (Turing → Hopper) and any newer driver — no per-CUDA-version
wheels, no compute-capability lock-in.

| Dimension | Support |
|---|---|
| GPU | Any NVIDIA (Turing / Ampere / Ada / Hopper / …) |
| Driver / CUDA | ≥ R525 / CUDA 12.0+. One wheel, all newer drivers |
| OS | Linux x86_64, glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL/Alma/Rocky 9+) |
| Python | CPython 3.10 – 3.13 |

Not yet: aarch64 (Jetson/Grace), glibc < 2.34, non-NVIDIA GPUs.

### Headless

Both backends run **fully headless** — no X server, nothing beyond the NVIDIA
driver:

- **GL** (default) uses **surfaceless EGL** → renders headless at full speed on a
  bare GPU server (cloud, cluster, container). The recommended path for training.
- **Vulkan** is also headless (shared device + exportable swapchain).

---

## 4. Your first render

The pattern is always: **step physics on GPU → copy the small transform arrays to
host `MjData` → render → get a `torch.cuda` tensor.**

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

mjw.step(M, d); wp.synchronize()
gx = d.geom_xpos.numpy()
gm = d.geom_xmat.numpy().reshape(N, mjm.ngeom, 9)
for i, h in enumerate(host):
    h.geom_xpos[:] = gx[i]; h.geom_xmat[:] = gm[i]

obs = r.render_batch(mjm, host, cam_id=0)   # (32, 256, 256, 4) uint8 torch.cuda
```

`obs` already lives on `cuda:0` — feed it straight into a CNN.

> **Always include a `<camera>` in the MJCF.** The renderer indexes camera 0; a
> scene with no camera reads invalid memory → CUDA illegal-access. Add e.g.
> `<camera name="cam0" pos="0 -3 1" xyaxes="1 0 0 0 0.3 0.95"/>`.

See [examples/minimal_render.py](../examples/minimal_render.py) for a runnable
demo.

---

## 5. Next steps

- Full API → [guide.md](guide.md)
- `train` vs `eval` presets → [guide.md → presets](guide.md#presets-train-vs-eval)
- Tune the look → [features.md](features.md)
- Common tasks → [cookbook.md](cookbook.md)
- CPU MuJoCo physics → [`mujofil`](https://github.com/tau-intelligence/MuJoCo-Filament)
