"""Mesh files in ComfyUI's own process (NumPy only, nothing to install): a connected MESH written for a runner, and a
runner's GLB read back as vertex and face arrays for a MESH output."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def write_obj(vertices, faces, path: Path) -> Path:
    """Vertices (N, 3) and triangles (M, 3, zero-based) as a Wavefront OBJ."""
    with open(path, "w", encoding="ascii") as handle:
        np.savetxt(handle, np.asarray(vertices, dtype=np.float64), fmt="v %.8g %.8g %.8g")
        np.savetxt(handle, np.asarray(faces, dtype=np.int64) + 1, fmt="f %d %d %d")
    return path


def mesh_arrays(mesh, index: int = 0):
    """Item `index` of a ComfyUI MESH batch as NumPy arrays, without the zero padding of a variable-size batch."""
    vertices, faces = mesh.vertices, mesh.faces
    if vertices.dim() == 2:
        vertices, faces = vertices[None], faces[None]
    vertex_counts = getattr(mesh, "vertex_counts", None)
    face_counts = getattr(mesh, "face_counts", None)
    n = int(vertex_counts[index]) if vertex_counts is not None else vertices.shape[1]
    m = int(face_counts[index]) if face_counts is not None else faces.shape[1]
    return (vertices[index, :n].detach().float().cpu().numpy(),
            faces[index, :m].detach().long().cpu().numpy())


def _quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF stores column-major
    matrix = np.eye(4)
    rotation = _quat_matrix(*node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    matrix[:3, :3] = rotation * np.asarray(node.get("scale", (1.0, 1.0, 1.0)))
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def read_glb(path) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle primitive of a GLB, node transforms applied, as (vertices (N, 3) float32, faces (M, 3) int64)."""
    data = Path(path).read_bytes()
    magic, _version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{path} is not a GLB file")
    offset, doc, binary = 12, None, b""
    while offset < len(data):
        size, kind = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8:offset + 8 + size]
        if kind == b"JSON":
            doc = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        offset += 8 + size
    if doc is None:
        raise ValueError(f"{path} has no glTF JSON chunk")

    def accessor(index):
        acc = doc["accessors"][index]
        view = doc["bufferViews"][acc["bufferView"]]
        dtype = np.dtype(_COMPONENTS[acc["componentType"]])
        width = _WIDTH[acc["type"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * width
        rows = np.ndarray((acc["count"], width), dtype=dtype, buffer=binary, offset=start,
                          strides=(stride, dtype.itemsize))
        return np.array(rows)

    vertices, faces, base = [], [], 0

    def visit(node_index, parent):
        nonlocal base
        node = doc["nodes"][node_index]
        world = parent @ _node_matrix(node)
        for primitive in doc["meshes"][node["mesh"]]["primitives"] if "mesh" in node else ():
            if primitive.get("mode", 4) != 4 or "POSITION" not in primitive["attributes"]:
                continue
            points = accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
            if "indices" in primitive:
                tris = accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64)
            else:
                tris = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
            vertices.append(points @ world[:3, :3].T + world[:3, 3])
            faces.append(tris + base)
            base += len(points)
        for child in node.get("children", ()):
            visit(child, world)

    scene = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {"nodes": range(len(doc["nodes"]))}
    for root in scene.get("nodes", ()):
        visit(root, np.eye(4))
    if not vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces)
