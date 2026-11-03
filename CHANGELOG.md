# Changelog

## 0.1.0 (2026-11-03)

First release.

- *LATO.2 Low-Poly Remesh*: a mesh (connected `MESH` or 3D file, a path, or a file from the input folder with an
  upload button) to a low-poly mesh of 200 to 5000 vertices; outputs a `MESH`, a 3D file and the GLB's path.
- Inner walls (TRELLIS.2's double surface, sealed interiors) are removed before remeshing.
- LATO.2's faces are rebuilt from its edge predictions into one closed surface with every face turned outward.
- *LATO.2 Setup Check*: whether the environment, code and weights are in place.
- `install_lato2.py` for Windows and Linux, with LATO.2's code and weights pinned to the tested versions; portable
  stand-ins for o_voxel, torch_scatter and flash-attn/xformers, and a CPU renderer for the conditioning view, so
  nothing is compiled.
- Example workflows, unit tests and a smoke test against a running ComfyUI.
