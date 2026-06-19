"""Optional asset tools for MuJoFil (install the ``usd`` extra).

Convert USD / GLB environments into a visual GLB + a MuJoCo collision MJCF that
the renderer can use. Requires ``pip install "mujofil-warp[usd]"`` (pulls usd-core +
trimesh + pillow).

CLI:
    python -m mujofil_warp.tools.usd_to_assets   scene.usd  --name kitchen
    python -m mujofil_warp.tools.glb_to_collision --glb env.glb --xform "16 floats" --name env
"""
