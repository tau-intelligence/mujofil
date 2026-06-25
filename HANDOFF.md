# mujofil-warp — Handoff

Status: **v0.1.7 published to PyPI** (2026-06-24). `pip install mujofil-warp==0.1.7`.
All regression gates green. This doc orients a new agent/developer: the three
codebases, where they live locally, how they fit together, how to build, and the
non-obvious gotchas.

---

## 1. The three codebases (local directories)

| Dir | What it is | Git |
|---|---|---|
| `/home/mumuksh/mujofil-warp` | **THE product.** GPU-native batched photoreal MuJoCo renderer (MJWarp physics + Filament PBR, zero-copy to `torch.cuda`). This is what ships to PyPI. | git, `github.com/tau-intelligence/mujofil-warp`, branch `master`, tag `v0.1.7` |
| `/home/mumuksh/MuJoCo-Filament` | **Sibling — `mujofil` (the "regular" renderer).** Same Filament PBR core but with STOCK MuJoCo (CPU) physics, no gl_Layer fork. Its `src/core/` is the UPSTREAM of the core this repo vendors at `native/vendor/core/`. Has its OWN `HANDOFF.md`. | git, `github.com/tau-intelligence/MuJoCo-Filament`, branch `main`; PyPI `mujofil` 0.1.7 |
| `/home/mumuksh/filament-build` | **Forked Google Filament v1.56.3.** Provides the renderer + the `matc`/`matinfo` tools. Forked for headless EGL + the `gl_Layer` layered/parallel-batch path. | git clone of `github.com/google/filament` at tag `v1.56.3`; fork edits are UNCOMMITTED working changes (captured as patches, see below) |
| `/home/mumuksh/Visual-Fidelity-Mujoco` | **Asset + scene authoring repo (VFM).** Photoreal GLB scenes (cafe/sponza/living/…), IBL maps, the `trailer.scenes.SCENES` dict (scene→GLB+xform+IBL+ambient), collision/asset pipeline scripts. Used by benchmarks; NOT shipped. | **NOT a git repo** (plain dir) |

`mujofil-warp` is the deliverable. `filament-build` is its rendering engine
(forked). `Visual-Fidelity-Mujoco` supplies test scenes/assets. `MuJoCo-Filament`
is the sibling CPU-physics renderer that shares the C++ render core.

> Stale copy to ignore: `/home/mumuksh/mujofil_pkg` is an old 0.1.0 snapshot of
> MuJoCo-Filament. The live regular-renderer working copy is `/home/mumuksh/MuJoCo-Filament`.

---

## 2. Environment / runtime

- **venv:** `/home/mumuksh/mujofil-warp/.venv` (numpy<2 pinned, torch 2.6+cu124,
  mujoco 3.9, mujoco_warp 3.9.0.1, trimesh, coacd). Activate: `source ~/mujofil-warp/.venv/bin/activate`.
- **Backend:** `MUJOFIL_WARP_BACKEND=gl` (surfaceless EGL OpenGL path). There is
  also a Vulkan path (`renderer_warp.cpp`) but GL is the primary/shipping one.
- **Hardware:** developed on RTX 4060 laptop. NVIDIA + CUDA required at runtime.

### CRITICAL gotchas (these will burn you)
1. **NEVER run `ulimit -v <N>` in the persistent terminal.** It can't be raised
   back and breaks torch+CUDA (looks like a renderer regression but isn't). To
   run GPU work under a clean limit, use a transient unit:
   ```bash
   systemd-run --user -p LimitAS=infinity --quiet --wait --pipe -- bash -c \
     'cd /home/mumuksh/mujofil-warp && source .venv/bin/activate && \
      PYTHONPATH=/home/mumuksh/mujofil-warp MUJOFIL_WARP_BACKEND=gl python <script>'
   ```
   (systemd-run loses cwd/PYTHONPATH — set them explicitly inside.) This is how
   every benchmark/test below is run.
2. **`trailer.scenes` lives at the VFM ROOT** (`~/Visual-Fidelity-Mujoco/trailer/scenes.py`).
   Add `~/Visual-Fidelity-Mujoco` to `sys.path` (not just `scripts/`) to import it.
3. Don't `pip install scipy` loosely into the venv — it pulls numpy 2.x and breaks
   the native ABI. numpy 1.26.4 + scipy 1.15.3 is the good combo.

---

## 3. Build workflow

### Native module (the common rebuild — after any `native/*.cpp|.h` edit)
```bash
cd ~/mujofil-warp/native && bash build_gl.sh          # -> _mujofil_warp_gl*.so
cp _mujofil_warp_gl*.so ../mujofil_warp/              # deploy into the package
```
Links against the prebuilt Filament libs at
`~/filament-build/out/release/filament/lib/x86_64/`.

### Filament fork (only when changing the renderer/shaders)
The fork = 5 files, captured as 2 patches in
`~/mujofil-warp/packaging/filament-patches/`:
- `0001-egl-desktop-gl-headless-fixes.patch` — `PlatformEGL.cpp`
- `0002-layered-batch-parallel-render.patch` — `OpenGLDriver.cpp`,
  `EngineEnums.h`, `CodeGenerator.cpp`, `shaders/src/main.vs`
  (the `gl_Layer = instance_index` routing for the single-draw layered path).

`~/filament-build` is a git checkout at tag `v1.56.3` with these as UNCOMMITTED
working changes. To rebuild after a shader/main.vs edit (shaders are baked into
`matc` at build time):
```bash
cd ~/filament-build/out/cmake-release && ninja matc          # rebuild matc only
# or `ninja filament matc` if you changed the renderer libs, then redeploy:
cp filament/libfilament.a ~/filament-build/out/release/filament/lib/x86_64/
```
Tools: `~/filament-build/out/cmake-release/tools/{matc/matc, matinfo/matinfo}`.
`matc -g` = unoptimised GLSL (keeps the gl_Layer fork un-stripped); `matinfo -g<i>`
dumps the i-th compiled GLSL shader (vertex shaders are the ones with
`gl_Layer`/`gl_Position`).

### Layered materials (only when changing `materials_layered_src/*.mat`)
```bash
cd ~/mujofil-warp
for m in default_pbr blend_pbr textured_pbr ground_grid; do
  ~/filament-build/out/cmake-release/tools/matc/matc -g -a opengl \
    -o mujofil_warp/materials_layered/$m.filamat materials_layered_src/$m.mat
done
```
Sources are in `materials_layered_src/`; compiled `.filamat` go into the package
at `mujofil_warp/materials_layered/`. **Both the standard `mujofil_warp/materials/`
and the forked `mujofil_warp/materials_layered/` are installed into the wheel**
(CMakeLists + pyproject sdist.include — this was a ship bug fixed in 0.1.7).

---

## 4. Architecture (what to know before touching the renderer)

Three render methods in `mujofil_warp/__init__.py` (all return `(…,4)` uint8
`torch.cuda`):
- `render()` — single view.
- `render_batch(model, datas, cam_id)` — **N worlds, multi-RT, ONE GPU sync.**
  Per-world egocentric cameras work here. Full Filament quality (gltfio ubershader
  PBR, post-processing, FILMIC, bloom). **This is the recommended egocentric
  photoreal path.**
- `render_batch_layered(model, datas, cam_id)` — **N worlds in ONE instanced draw**
  via the forked `gl_Layer` array-texture path. Max throughput, lower fidelity
  (post-processing off; uses our 4 simplified `materials_layered` materials).
  `cam_id<0` = shared camera; `cam_id>=0` = egocentric (view-folding).

### Key 0.1.7 facts (this session's work)
- **`train` preset is the headline win.** `render_batch` was wasting compute on
  per-VIEW effect passes (MSAA4 supersample-resolve, SSAO, SSR — each run N times).
  `preset="train"` (and `fast`/`raw`) now also disable **SSR** (the dominant hidden
  cost; was the missing piece). Result: ~4× faster at low res (cafe 128px N=64:
  1487→6059 cam/s) while keeping FULL gltfio PBR + textures + FILMIC. Use:
  `WarpRenderer(width=128, height=128, batch_size=N, preset="train")`.
- **`load_glb_layered(path, xform16)`** ingests a photoreal GLB into the instanced
  layered path (parses with trimesh, bakes to world, carries albedo+normal+MR+
  emissive maps with UV-aligned tangents). Lets egocentric GLB ride the single
  instanced draw. NOTE on fidelity: the layered objects pass is post-processing-OFF,
  so it tone-maps in Python (deterministic FILMIC + percentile auto-exposure, no
  per-scene tuning). The objects render target is **FLOAT16 (RGBA16F) HDR** — this
  was the fix that killed the dark-region colour banding (8-bit linear posterised).
- **Honest verdict (important):** for egocentric photoreal, `render_batch` +
  `preset="train"` BEATS the layered path on BOTH quality and speed at typical RL
  res. The layered path is a niche max-throughput/lower-fidelity option. The
  theoretical "N views at M total cost" is impossible for INDEPENDENT cameras
  (each shades its own pixels = irreducibly N× fragment work; only a SHARED camera
  lets you render-once-reuse).

---

## 5. Validation / regression (run before any release)

All via the `systemd-run` wrapper above, with the venv + `PYTHONPATH` + `MUJOFIL_WARP_BACKEND=gl`:
```bash
python benchmarks/smoke_test.py            # all render paths (incl float16 layered)
python benchmarks/tile_check.py            # layered routing CI gate -> 9/9
python benchmarks/adversarial_layered.py   # layered robustness -> 19/19
python benchmarks/ego_geomvary.py          # native egocentric correctness -> 6/6
python test_batch.py                       # default multi-RT path -> BATCH OK
python benchmarks/reload_stress.py         # leak/double-free on reload
```
Throughput/quality demos (write to `out/`):
`preset_check.py`, `batch_fast.py`, `ego_glb_layered.py`, `env_showcase.py`,
`ego_vs_baselines.py` (vs vanilla MuJoCo + MJWarp).

---

## 6. Persistent notes / deeper history

Detailed engineering notes (multi-session) live in agent **repo memory**:
`/memories/repo/renderer.md` (architecture, the gl_Layer fork, egocentric/view-
folding, the float16-RT + tonemap saga, the train-preset finding, ship bugs) and
`/memories/repo/project-overview.md` (release history). Read these first for the
"why" behind any non-obvious code.

The mujofil-warp git history is also descriptive — `git log` on `master`, esp. the
`v0.1.7` and `v0.1.6` commits.

---

## 7. Open threads / next ideas (not blocking)

- The layered path's quality gap vs render_batch is inherent (post-processing off,
  simplified materials). If true single-draw photoreal egocentric is ever needed,
  it would require porting the gltfio ubershader features (normal/MR/emissive maps
  are done; missing: KHR transmission/refraction, screen-space post) into the
  layered materials — large, diminishing returns given `train` already wins.
- A quality-preserving render_batch speedup that DOES exist: merge GLB meshes that
  share a material to cut the per-view draw-call count (helps few-material scenes;
  limited for sponza's 102 unique materials). Not yet done.
- VFM is not a git repo — if its asset pipeline matters to the other agent,
  consider initialising one.
