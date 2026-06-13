"""Shared scene for the A/B comparison (kept tiny so workers stay lean)."""
from __future__ import annotations


def build_xml() -> str:
    """Material spheres + a steel box on a collision-only floor (group=3 so
    MJWarp physics has a floor but mujofil draws the warehouse GLB floor)."""
    specs = [
        ("chrome",  "0.95 0.95 0.97 1", 1.0, 0.06),
        ("gold",    "1.0 0.78 0.34 1",  1.0, 0.20),
        ("copper",  "0.95 0.55 0.35 1", 1.0, 0.38),
        ("plastic", "0.20 0.45 0.85 1", 0.0, 0.30),
        ("rubber",  "0.85 0.25 0.22 1", 0.0, 0.85),
    ]
    mats, bodies = [], []
    n = len(specs)
    for i, (name, rgba, met, rough) in enumerate(specs):
        mats.append(
            f'<material name="{name}" rgba="{rgba}" metallic="{met}" roughness="{rough}"/>')
        x = (i - (n - 1) / 2.0) * 0.62
        bodies.append(
            f'<body name="b{i}" pos="{x:.3f} 0 0.30"><freejoint/>'
            f'<geom type="sphere" size="0.24" material="{name}"/></body>')
    mats.append('<material name="steel" rgba="0.8 0.82 0.85 1" metallic="1.0" roughness="0.25"/>')
    bodies.append('<body name="box" pos="0 0.7 0.32"><freejoint/>'
                  '<geom type="box" size="0.22 0.22 0.22" material="steel"/></body>')
    return f"""
<mujoco model="ab_scene">
  <option timestep="0.004"/>
  <asset>
    {''.join(mats)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" group="3"/>
    {''.join(bodies)}
    <camera name="cam0" pos="0 -3.2 1.25" xyaxes="1 0 0 0 0.36 0.93"/>
  </worldbody>
</mujoco>"""
