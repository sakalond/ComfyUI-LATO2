"""ComfyUI nodes for LATO.2 low-poly re-meshing.

LATO.2 Low-Poly Remesh takes a connected MESH (native TRELLIS.2, Hunyuan3D) or 3D file (Load 3D), a path, or a mesh
picked or uploaded from the input folder, runs LATO.2 in its own environment (runtime.py) and returns a MESH
(Save 3D Model and the other mesh nodes), a 3D file (Preview 3D) and the GLB's path relative to the output folder.
"""
from __future__ import annotations

import re
import types as _types
import uuid
from pathlib import Path
from typing import Any

try:
    from .mesh_io import mesh_arrays, read_glb, write_obj
    from .runtime import run, status_text
except ImportError:  # pragma: no cover - direct `import nodes` (tests)
    from mesh_io import mesh_arrays, read_glb, write_obj
    from runtime import run, status_text

try:
    from comfy_api.latest import Types  # type: ignore  # MESH and File3D, ComfyUI 0.3.6x and later
except Exception:  # pragma: no cover - older ComfyUI or outside it
    Types = None

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - outside ComfyUI
    folder_paths = None

try:
    from comfy.utils import ProgressBar  # type: ignore
except Exception:  # pragma: no cover
    ProgressBar = None

try:
    from server import PromptServer  # type: ignore
except Exception:  # pragma: no cover
    PromptServer = None

CATEGORY = "3d/LATO.2"
SUBFOLDER = "lato2"
PROGRESS_EVENT = "lato2_progress"
MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".stl")
# A connectable input of several types, as ComfyUI's own MultiType inputs declare them.
MESH_INPUT_TYPES = "MESH,FILE_3D,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_PLY,FILE_3D_STL"


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


class _Progress:
    """ComfyUI's progress bar plus a `lato2_progress` event that carries the stage name."""

    def __init__(self, node_id: Any, run_id: str):
        self.node_id = str(_scalar(node_id) or "")
        self.run_id = run_id
        self.value = 0
        try:
            self._bar = ProgressBar(100) if ProgressBar is not None else None
        except Exception:
            self._bar = None

    def __call__(self, value: float, stage: str) -> None:
        value = max(self.value, min(100, int(round(value))))
        self.value = value
        if self._bar is not None:
            try:
                self._bar.update_absolute(value, 100)
            except Exception:
                pass
        if PromptServer is not None:
            try:
                PromptServer.instance.send_sync(PROGRESS_EVENT, {"node": self.node_id, "run_id": self.run_id,
                                                                 "value": value, "max": 100, "stage": stage})
            except Exception:
                pass


def _input_dir() -> Path:
    return Path(folder_paths.get_input_directory()) if folder_paths is not None else Path.cwd() / "input"


def _output_dir() -> Path:
    return Path(folder_paths.get_output_directory()) if folder_paths is not None else Path.cwd() / "output"


def _work_dir() -> Path:
    """Where a connected mesh is written for the runner (ComfyUI's temp folder, cleared when it starts)."""
    if folder_paths is not None:
        return Path(folder_paths.get_temp_directory()) / SUBFOLDER
    import tempfile
    return Path(tempfile.gettempdir()) / SUBFOLDER


def mesh_files() -> list[str]:
    """Meshes in ComfyUI's input folder (including input/3d, where Load 3D and the upload button put them)."""
    base = _input_dir()
    if not base.is_dir():
        return []
    return sorted(p.relative_to(base).as_posix() for p in base.rglob("*")
                  if p.is_file() and p.suffix.lower() in MESH_EXTENSIONS)


def resolve(value: str, what: str) -> Path:
    """An absolute path, or one relative to ComfyUI's input or output folder."""
    raw = str(_scalar(value) or "").strip().strip('"')
    if not raw:
        raise ValueError(f"{what} is empty")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [_input_dir() / raw, _output_dir() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{what} not found: {raw}")


def write_connected(value: Any, work: Path) -> Path:
    """A connected MESH or 3D file as a file the runner can read."""
    work.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        return resolve(value, "mesh")
    if hasattr(value, "save_to"):  # File3D (Load 3D, Create 3D File (from Mesh), ...)
        fmt = (getattr(value, "format", "") or "glb").lstrip(".").lower()
        path = work / f"input_{uuid.uuid4().hex[:8]}.{fmt}"
        value.save_to(str(path))
        return path
    if hasattr(value, "vertices") and hasattr(value, "faces"):  # MESH: geometry only
        vertices, faces = mesh_arrays(value)
        if value.vertices.dim() == 3 and value.vertices.shape[0] > 1:
            print(f"[LATO.2] the connected MESH holds {value.vertices.shape[0]} meshes; using the first", flush=True)
        return write_obj(vertices, faces, work / f"input_{uuid.uuid4().hex[:8]}.obj")
    raise TypeError(f"cannot read a mesh from {type(value).__name__}")


def pick_mesh(work: Path, connected=None, path="", model_file="none") -> tuple[Path, bool]:
    """The input mesh from, in this order: a connection, a path, a file picked from the input folder. Returns the
    path and whether it is a temporary file of this run."""
    if connected is not None:
        return write_connected(connected, work), not isinstance(connected, str)
    if str(_scalar(path) or "").strip():
        return resolve(path, "input_path"), False
    if model_file and model_file != "none":
        return resolve(model_file, "model_file"), False
    raise ValueError("connect a mesh, pick a model_file or give input_path")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_output_dir().resolve()).as_posix()
    except ValueError:
        return str(path)


def _output_path(prefix: str) -> Path:
    folder = _output_dir() / SUBFOLDER
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix or "")).strip("._")[:80] or "lato2"
    return folder / f"{slug}_{uuid.uuid4().hex[:8]}.glb"


def _mesh_output(path: Path):
    vertices, faces = read_glb(path)
    if Types is not None:
        import torch
        return Types.MESH(vertices=torch.from_numpy(vertices)[None], faces=torch.from_numpy(faces)[None])
    return _types.SimpleNamespace(vertices=vertices, faces=faces)


def _file_output(path: Path):
    return Types.File3D(str(path), "glb") if Types is not None else _relative(path)


class LATO2Remesh:
    """Re-mesh a mesh into a low-poly one with generated, artist-like topology (LATO.2): one closed surface, faces
    turned outward, at the input's position and size."""

    CATEGORY = CATEGORY
    DESCRIPTION = ("Rebuilds a mesh as a low-poly mesh with generated topology (LATO.2). Inner walls (as in TRELLIS.2 "
                   "meshes) are removed first; the result is one closed surface at the input's position and size. "
                   "Geometry only: textures are not carried over.")
    RETURN_TYPES = ("MESH", "FILE_3D_GLB", "STRING")
    RETURN_NAMES = ("mesh", "model_3d", "glb_path")
    OUTPUT_TOOLTIPS = ("The low-poly mesh (geometry only), for Save 3D Model and the other mesh nodes",
                       "The GLB, for Preview 3D or Save 3D", "The GLB's path relative to the output folder")
    FUNCTION = "remesh"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vertex_count": ("INT", {"default": 2000, "min": 200, "max": 5000, "step": 50,
                                         "tooltip": "Vertices LATO.2 places (the result has about this many)"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "mesh": (MESH_INPUT_TYPES, {"tooltip": "A mesh or 3D file to re-mesh (overrides the fields below)"}),
                "model_file": (["none"] + mesh_files(), {
                    "default": "none", "file_upload": True,
                    "tooltip": "A mesh in ComfyUI's input folder (.glb .gltf .obj .ply .stl); 'upload mesh' adds one"}),
                "input_path": ("STRING", {"default": "", "tooltip": "A mesh path: absolute, or in ComfyUI's input or "
                                                                    "output folder (overrides model_file)"}),
                "filename_prefix": ("STRING", {"default": "lato2",
                                               "tooltip": "Name of the GLB in output/lato2"}),
                "vflow_steps": ("INT", {"default": 24, "min": 4, "max": 100, "advanced": True,
                                        "tooltip": "Steps of the vertex flow"}),
                "tflow_steps": ("INT", {"default": 50, "min": 4, "max": 200, "advanced": True,
                                        "tooltip": "Steps of the topology flow"}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 15.0, "step": 0.1, "advanced": True,
                                  "tooltip": "Guidance of the vertex flow by the input's rendered view"}),
                "remove_inner_walls": ("BOOLEAN", {"default": True, "advanced": True,
                                                   "tooltip": "Drop faces that empty space from outside cannot reach "
                                                              "(inner walls, sealed interiors) before re-meshing"}),
                "rebuild_surface": ("BOOLEAN", {"default": True, "advanced": True,
                                                "tooltip": "Rebuild the faces from LATO.2's edge predictions as one "
                                                           "closed surface (off: LATO.2's raw faces, which overlap "
                                                           "and leave holes)"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def remesh(self, vertex_count, seed, mesh=None, model_file="none", input_path="", filename_prefix="lato2",
               vflow_steps=24, tflow_steps=50, cfg=3.0, remove_inner_walls=True, rebuild_surface=True,
               unique_id=None):
        output = _output_path(filename_prefix)
        source, temporary = pick_mesh(_work_dir(), mesh, input_path, model_file)
        args = ["--input", str(source), "--output", str(output), "--vert_num", str(int(vertex_count)),
                "--seed", str(int(seed)), "--vflow_steps", str(int(vflow_steps)),
                "--tflow_steps", str(int(tflow_steps)), "--cfg", str(float(cfg)),
                "--solidify" if remove_inner_walls else "--no-solidify",
                "--surface" if rebuild_surface else "--no-surface"]
        try:
            run(args, on_progress=_Progress(unique_id, output.stem))
        finally:
            if temporary:
                source.unlink(missing_ok=True)
        return {
            "ui": {"files": [{"filename": output.name, "subfolder": SUBFOLDER, "type": "output"}],
                   "text": [_relative(output)]},
            "result": (_mesh_output(output), _file_output(output), _relative(output)),
        }


class LATO2SetupCheck:
    """Report whether LATO.2 is set up and where it looks for its environment, code and weights."""

    CATEGORY = CATEGORY
    DESCRIPTION = "Shows whether LATO.2's environment, code and weights are in place, and how to install them."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "check"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def check(self):
        text = status_text()
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "LATO2Remesh": LATO2Remesh,
    "LATO2SetupCheck": LATO2SetupCheck,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LATO2Remesh": "LATO.2 Low-Poly Remesh",
    "LATO2SetupCheck": "LATO.2 Setup Check",
}
