# mujofil-warp Documentation

**Photoreal PBR rendering for GPU-resident MuJoCo ([MJWarp](https://github.com/google-deepmind/mujoco_warp)),
zero-copy to PyTorch.** MJWarp simulates thousands of parallel worlds on the GPU;
`mujofil-warp` renders them with [Google Filament](https://github.com/google/filament)'s
physically-based renderer and delivers each frame **straight to PyTorch as a
`torch.cuda` tensor — no GPU→CPU→GPU bounce**.

> ### 🖥️ Running normal CPU MuJoCo instead?
> If you step physics with `mujoco.mj_step` on the CPU (one env or a CPU
> vector-env) and just want photoreal **NumPy** frames, use the CPU edition:
> **→ [`mujofil`](https://github.com/tau-intelligence/MuJoCo-Filament)**
> ([its docs](https://github.com/tau-intelligence/MuJoCo-Filament/tree/main/docs)).
>
> Same Filament renderer, same materials/IBL/shadows — it differs only in *where
> physics runs* (CPU) and *what the pixels come back as* (NumPy).

---

## Start here

| Guide | What it covers |
|---|---|
| [getting-started.md](getting-started.md) | Install, the NVIDIA-driver requirement, and your first zero-copy render. |
| [guide.md](guide.md) | The full API: the MJWarp→render loop, `WarpRenderer`, presets (`train`/`eval`), quality toggles, batched zero-copy, GL vs Vulkan backends, headless operation, profiling. |
| [features.md](features.md) | A single reference table of **every** fidelity feature (PBR, IBL, SSAO, shadows, bloom, AA, tone mapping) — what it does, its cost, and which toggle controls it. |
| [cookbook.md](cookbook.md) | Task-oriented recipes (vision-RL observations, `train` vs `eval`, drop objects into a photoreal warehouse) and a troubleshooting table. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The design + phased zero-copy integration plan. |

---

## The 30-second mental model

```mermaid
flowchart LR
  D[mujoco_warp on GPU] -->|tiny geom-transform copy| E[WarpRenderer]
  E -->|Filament PBR, zero-copy| F[torch.cuda N,H,W,4]
```

- MJWarp steps physics on the GPU; only a small transform array crosses to the
  host. **Pixels never leave the GPU.**
- Frames come back as **`torch.cuda`** `(N, H, W, 4)` uint8 — already on the GPU,
  feed straight into a CNN (no `.cuda()` copy).
- A neutral **studio HDR** is available so even a bare `.xml` gets real
  image-based lighting and reflections.

---

## Why not MJWarp's built-in renderer?

MJWarp ships a batch renderer, but it's a deliberately **low-fidelity single-hit
raycaster**: flat Lambertian, no PBR / IBL / reflections, and it **cannot load
GLB** environments. `mujofil-warp` is the photoreal alternative for the same
GPU-resident physics — and at small-to-moderate batch sizes it's also *faster*
(see [Performance](guide.md#performance)).

---

## Requirements at a glance

- An **NVIDIA GPU + driver ≥ R525 (CUDA 12.0+)** — the only hard runtime
  requirement. No CUDA toolkit, no Filament, no `mujofil` to install.
- `mujoco_warp` and a CUDA `torch` in your environment.
- Linux x86-64, glibc ≥ 2.34, Python 3.10–3.13. Both backends run **fully
  headless** (no X server).

> Documented for `mujofil-warp` 0.1.2.
