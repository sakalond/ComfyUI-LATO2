"""End-to-end check against a running ComfyUI with this node installed (needs the GPU and the installed model):

    python tests/smoke_comfyui.py --server 127.0.0.1:8188 --mesh some_mesh.glb [--vertices 800]

Uploads the mesh to input/3d (as the "upload mesh" button does), then queues one graph:
    LATO.2 (picked file) -> Get Mesh Info, Save 3D Model (MESH output), Preview 3D (FILE_3D output)
    LATO.2 fed the first one's MESH output, and another fed its FILE_3D output
and checks that every node ran and every GLB written is one closed mesh (no open or non-manifold edges).
Not collected by pytest (it needs a server); the unit tests run without one.
"""
import argparse
import json
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh_io import read_glb  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--vertices", type=int, default=800)
    args = parser.parse_args()
    server = args.server

    def get(path):
        with urllib.request.urlopen(f"http://{server}{path}", timeout=60) as response:
            return json.loads(response.read())

    def post(path, body, content_type="application/json"):
        request = urllib.request.Request(f"http://{server}{path}", data=body, method="POST",
                                         headers={"Content-Type": content_type})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())

    def upload(path: Path):
        boundary = uuid.uuid4().hex
        parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
                 for name, value in (("type", "input"), ("subfolder", "3d"), ("overwrite", "true"))]
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
                     f"Content-Type: application/octet-stream\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return post("/upload/image", b"".join(parts), f"multipart/form-data; boundary={boundary}")

    def closed(subfolder, filename):
        query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
        with urllib.request.urlopen(f"http://{server}/view?{query}", timeout=300) as response:
            local = Path(tempfile.gettempdir()) / f"lato2_smoke_{filename}"
            local.write_bytes(response.read())
        import numpy as np
        vertices, faces = read_glb(local)
        _, inverse = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
        faces = inverse.reshape(-1)[faces]
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return {"faces": len(faces), "open_edges": int((counts == 1).sum()),
                "non_manifold_edges": int((counts > 2).sum())}

    status = get("/object_info/LATO2SetupCheck")
    assert "LATO2SetupCheck" in status, "the LATO2 nodes are not registered"
    uploaded = upload(args.mesh)
    picked = f"{uploaded['subfolder']}/{uploaded['name']}"
    options = get("/object_info/LATO2Remesh")["LATO2Remesh"]["input"]["optional"]["model_file"][0]
    print(f"[smoke] uploaded {picked}; in the picker: {picked in options}", flush=True)
    half = max(200, args.vertices // 2)

    def lato(**inputs):
        return {"class_type": "LATO2Remesh", "inputs": {"vertex_count": args.vertices, "seed": 42, **inputs}}

    prompt = {
        "1": lato(model_file=picked, filename_prefix="smoke"),
        "2": {"class_type": "GetMeshInfo", "inputs": {"mesh": ["1", 0]}},
        "3": {"class_type": "SaveGLB", "inputs": {"mesh": ["1", 0], "filename_prefix": "3d/lato2_smoke"}},
        "4": {"class_type": "Preview3D", "inputs": {"model_file": ["1", 1]}},
        "5": lato(mesh=["1", 0], vertex_count=half, filename_prefix="smoke_from_mesh"),
        "6": lato(mesh=["1", 1], vertex_count=half, filename_prefix="smoke_from_file"),
        "7": {"class_type": "LATO2SetupCheck", "inputs": {}},
    }
    started = time.time()
    prompt_id = post("/prompt", json.dumps({"prompt": prompt, "client_id": uuid.uuid4().hex}).encode())["prompt_id"]
    while True:
        entry = get(f"/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None:
            break
        if entry and entry.get("status", {}).get("status_str") == "error":
            break
        time.sleep(5)
    state = entry["status"].get("status_str")
    print(f"[smoke] {state} in {time.time() - started:.0f} s", flush=True)
    for message in entry["status"].get("messages", []):
        if message[0] == "execution_error":
            print("[smoke] error:", json.dumps(message[1])[:1500], flush=True)
    outputs = entry.get("outputs", {})
    meshes = {}
    for node in ("1", "5", "6"):
        for ref in outputs.get(node, {}).get("files", []):
            meshes[f"LATO.2 node {node}"] = closed(ref["subfolder"], ref["filename"])
    for ref in outputs.get("3", {}).get("3d", []):
        meshes["Save 3D Model"] = closed(ref["subfolder"], ref["filename"])
    print("[smoke]", json.dumps(meshes), flush=True)
    print("[smoke] setup check:", outputs.get("7", {}).get("text", ["?"])[0].splitlines()[0], flush=True)
    ok = state == "success" and len(meshes) == 4 and all(
        m["open_edges"] == 0 and m["non_manifold_edges"] == 0 for m in meshes.values())
    print("[smoke] ok" if ok else "[smoke] FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
