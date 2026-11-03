"""The node outside ComfyUI: its inputs, where it takes the mesh from, mesh files in and out, and the settings."""
import json
import types

import numpy as np
import pytest

import mesh_io
import nodes
import runtime


def test_inputs_and_outputs():
    spec = nodes.LATO2Remesh.INPUT_TYPES()
    assert list(spec["required"]) == ["vertex_count", "seed"]
    assert spec["optional"]["mesh"][0].split(",")[:2] == ["MESH", "FILE_3D"]
    assert spec["optional"]["model_file"][1]["file_upload"] is True
    assert nodes.LATO2Remesh.RETURN_TYPES == ("MESH", "FILE_3D_GLB", "STRING")


def test_input_order_connection_then_path_then_picker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input" / "3d").mkdir(parents=True)
    picked = tmp_path / "input" / "3d" / "picked.obj"
    typed = tmp_path / "input" / "typed.obj"
    for path in (picked, typed):
        mesh_io.write_obj(np.eye(3), [[0, 1, 2]], path)
    assert nodes.mesh_files() == ["3d/picked.obj", "typed.obj"]
    work = tmp_path / "work"
    assert nodes.pick_mesh(work, None, "", "3d/picked.obj") == (picked.resolve(), False)
    assert nodes.pick_mesh(work, None, "typed.obj", "3d/picked.obj") == (typed.resolve(), False)
    with pytest.raises(ValueError):
        nodes.pick_mesh(work, None, "", "none")


def test_connected_mesh_is_written_for_the_runner(tmp_path):
    torch = pytest.importorskip("torch")
    vertices = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]]])
    faces = torch.tensor([[[0, 1, 2], [0, 0, 0]]])
    # A padded batch item: three real vertices and one real face.
    mesh = types.SimpleNamespace(vertices=vertices, faces=faces, vertex_counts=torch.tensor([3]),
                                 face_counts=torch.tensor([1]))
    path, temporary = nodes.pick_mesh(tmp_path, mesh)
    assert temporary and path.suffix == ".obj"
    lines = path.read_text().splitlines()
    assert sum(line.startswith("v ") for line in lines) == 3
    assert [line for line in lines if line.startswith("f ")] == ["f 1 2 3"]


def test_read_glb_matches_trimesh_with_node_transforms(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.Scene()
    box = trimesh.creation.box((1, 2, 3))
    scene.add_geometry(box, transform=trimesh.transformations.translation_matrix([5, 0, 0]))
    scene.add_geometry(trimesh.creation.icosphere(1))
    path = tmp_path / "scene.glb"
    scene.export(path)
    vertices, faces = mesh_io.read_glb(path)
    reference = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    assert len(faces) == len(reference.faces)
    assert np.allclose(np.sort(vertices, axis=0), np.sort(np.asarray(reference.vertices, np.float32), axis=0),
                       atol=1e-5)


def test_settings_defaults_then_file_then_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "CONFIG_PATH", tmp_path / "lato2_config.json")
    monkeypatch.delenv("LATO2_CKPT", raising=False)
    defaults = runtime.config()
    assert defaults["repo"].endswith("LATO.2") and defaults["ckpt"].endswith("lato2")
    (tmp_path / "lato2_config.json").write_text(json.dumps({"ckpt": "from-file"}))
    assert runtime.config()["ckpt"] == "from-file"
    monkeypatch.setenv("LATO2_CKPT", "from-env")
    assert runtime.config()["ckpt"] == "from-env"
