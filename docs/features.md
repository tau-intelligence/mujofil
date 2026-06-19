# Feature Reference

Every fidelity feature in the Filament core, what it does, what it costs, and the
toggle that controls it.

> The **same renderer** powers the CPU edition
> [`mujofil`](https://github.com/tau-intelligence/MuJoCo-Filament); the option
> names differ slightly there (e.g. `enable_ssao` on a `RenderConfig` instead of
> the `ssao` toggle). This page documents the `mujofil-warp` toggles (used via
> `WarpRenderer(...)` kwargs or `make_config`).

| Feature | Toggle | Cost | Default |
|---|---|---|---|
| [PBR materials](#pbr-materials) | always on | baseline | on |
| [Image-based lighting](#image-based-lighting-ibl) | `load_ibl(...)`, `set_ambient_intensity(...)` | low | available |
| [SSAO](#ssao-ambient-occlusion) | `ssao`, `ssao_quality`, `ssao_ssct` | **highest** | on (off in `train`) |
| [Shadows](#shadows) | `shadows` | medium | on |
| [Bloom](#bloom) | `bloom` | low | off |
| [Anti-aliasing](#anti-aliasing) | `msaa`/`msaa_samples`, `fxaa` | varies | MSAA 4× on |
| [Tone mapping & exposure](#tone-mapping--exposure) | `tone_mapping`, `exposure`, `dithering` | free | filmic, exposure 0.0 |
| [Backend](#backend) | `MUJOFIL_WARP_BACKEND` | — | GL |

---

## PBR materials

Physically-based metalness/roughness shading — the foundation of the photoreal
look. Reads MuJoCo material properties directly:

- `material rgba` → base color
- `material metallic` → metalness (0 = dielectric, 1 = metal)
- `material roughness` → microsurface roughness (0 = mirror, 1 = matte)
- `material emission` → emissive
- `material reflectance` → glossier, more reflective surface
- 2D and cubemap **textures**

Define materials in your MJCF and they render correctly with no extra code:

```xml
<asset>
  <material name="chrome" rgba="0.95 0.95 0.97 1" metallic="1.0" roughness="0.08"/>
  <material name="gold"   rgba="1.0 0.78 0.34 1"  metallic="1.0" roughness="0.22"/>
</asset>
```

> **Metals need an environment to reflect.** A metallic surface with no IBL and
> nothing around it renders dark — that's physically correct. Load an IBL or a
> GLB environment so metals have something to reflect. This is exactly where the
> PBR renderer beats MJWarp's flat raycaster.

---

## Image-based lighting (IBL)

A pre-filtered HDR environment provides realistic ambient light **and** specular
reflections — the single biggest fidelity lever.

```python
r.load_ibl("myenv_ibl.ktx", "myenv_skybox.ktx")
r.set_ambient_intensity(6500.0)
```

Convert a `.hdr` (e.g. [Poly Haven](https://polyhaven.com), CC0) to Filament KTX
with `cmgen`:

```bash
cmgen --type=ktx --format=ktx --size=256 --deploy=myenv myenv.hdr
# -> myenv/myenv_ibl.ktx + myenv/myenv_skybox.ktx
```

> Load an IBL **before** `set_ambient_intensity` — ambient only *scales* an
> existing IBL.

---

## SSAO (ambient occlusion)

Screen-space ambient occlusion darkens crevices and contact points. It is the
**largest single render cost** — turning it off is roughly **2× faster**.

- `ssao` — on/off (off in the `train` preset).
- `ssao_quality` — `low`/`medium`/`high`/`ultra` (or 0–3). Affects the *look*
  more than the speed; the SSAO **pass** is the cost, not its quality.
- `ssao_ssct` — cone tracing for extra contact shadows (small extra cost).

For RL observations at 128px, SSAO subtlety is imperceptible to a CNN, so `train`
turns it **off**.

---

## Shadows

`shadows` toggles soft directional shadow maps cast by lights with
`cast_shadows=True`. Only directional (sun) and spot lights cast shadows. High
ambient/IBL washes shadows out — lower ambient and use an oblique key light.

---

## Bloom

`bloom` adds HDR glow bleeding from bright/emissive surfaces. Cheap-ish, off by
default; on in the `ultra` preset. Turn on for cinematics or glowing fixtures.

---

## Anti-aliasing

| Method | Toggle | Notes |
|---|---|---|
| **MSAA** | `msaa`, `msaa_samples` (2/4/8) | Multisample AA, part of the render path. On (4×) by default; off in `train`. |
| **FXAA** | `fxaa` | Cheap, slightly blurry — alternative to MSAA. |

> **RL tip:** keep AA settings consistent between training and evaluation so the
> pixel distribution your policy sees doesn't shift. `train` drops MSAA (edges are
> imperceptible at 128px); use `eval` for human-watched video.

---

## Tone mapping & exposure

- `tone_mapping=True` → FILMIC tone curve; `False` → linear.
- `exposure` → brightness before tone mapping (default 0.0). Raise for dark
  scenes, lower if bright surfaces blow out.
- `dithering=True` → temporal dithering to avoid banding.

Typical exposure starting points: studio/neutral ~0.0–0.8, dark interior 1.2–1.6.

---

## Backend

`MUJOFIL_WARP_BACKEND=gl` (default) is the OpenGL single-sync path — fastest, with
sync cost constant in batch size, fully headless via surfaceless EGL.
`MUJOFIL_WARP_BACKEND=vulkan` is the shared-device path; also headless but its
sync cost grows with N (2-frame in-flight cap). See
[guide → backends](guide.md#headless--backends).

---

## Quick "make it look good" defaults

| Goal | Settings |
|---|---|
| Eval video / cinematic | `preset="eval"` (or `"ultra"` + bloom), higher res |
| Fast RL observations | `preset="train"` (SSAO + MSAA off), 128px |
| Metals that pop | load a rich IBL or a GLB environment to reflect |
| Dark interior | `load_ibl` + lower ambient + oblique key + higher exposure |

See [cookbook.md](cookbook.md) for full, copy-pasteable recipes.
