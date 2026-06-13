# mujofil-warp

**Photoreal PBR rendering for GPU-resident MuJoCo (MJWarp).**

MJWarp ([google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp))
gives you thousands of parallel MuJoCo worlds simulated entirely on the GPU. Its
built-in batch renderer is a deliberately **low-fidelity single-hit raycaster**
(flat Lambertian shading, basic lights, no PBR / IBL / global illumination).

`mujofil-warp` aims to pair MJWarp's GPU-resident physics with
[Google Filament](https://github.com/google/filament)'s **physically-based
renderer** (PBR materials, image-based lighting, soft shadows) — keeping data on
the GPU to avoid the CPU round-trip that kills throughput.

> Status: **early R&D / feasibility.** This project is a separate, exploratory
> effort ("Option B") that builds on the CPU-MuJoCo `mujofil` renderer. The
> standalone CPU-MuJoCo `mujofil` package remains available for users who want a
> high-fidelity renderer for vector-env vision RL without MJWarp.

## Why

| | MJWarp built-in renderer | mujofil-warp (goal) |
|---|---|---|
| Physics | GPU-resident (Warp) | GPU-resident (Warp) — reused |
| Shading | single-hit Lambertian | full **PBR** (metalness/roughness) |
| Lighting | point/dir lights | **IBL + GI + soft shadows** |
| Materials | flat + texture | **physically based** |
| Use case | RL throughput, low fidelity | **sim-to-real / synthetic data / RL with photoreal pixels** |

The target user trains or generates data where the **reality gap** matters
(perception, sim-to-real, domain randomization with realistic lighting) but still
wants MJWarp's parallel GPU throughput.

## Layout

```
scripts/smoke_mjwarp.py   # baseline: MJWarp GPU physics + its low-fi raycaster
docs/ARCHITECTURE.md      # feasibility analysis + phased integration plan
out/                      # sample renders (gitignored)
```

## Baseline (RTX 4060 Laptop, 8 GiB)

Simple 3-geom scene, 16 worlds:
- **Physics:** ~64,000 env-steps/s (CUDA-graph, post-compile)
- **MJWarp render:** ~670 cam-frames/s @ 256² (single-hit raycaster)
- **Fidelity:** flat Lambertian (see `out/mjwarp_world0.png`)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the integration plan and the
key open risk (Vulkan↔CUDA external-memory interop with the prebuilt Filament).

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mujoco-warp pillow
python scripts/smoke_mjwarp.py --nworld 16 --res 256
```
