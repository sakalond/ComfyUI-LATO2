"""Set up LATO.2 for this node, on Windows or Linux (NVIDIA GPU, git, internet):

    python install_lato2.py [--models-dir DIR] [--only repo,env,models,config]

  repo     LATO.2's code into vendor/LATO.2 (pinned to the tested commit)
  env      envs/lato2: Python 3.12, PyTorch 2.8 (CUDA 12.8), spconv and xformers wheels (about 8 GB). Nothing is
           compiled: o_voxel, torch_scatter and the conditioning-view renderer have portable stand-ins
           (runners/portable.py, runners/white_model_render.py).
  models   LATO.2's weights (3.4 GB, pinned revision) and DINOv2 (1.1 GB) into MODELS_DIR, by default ComfyUI's
           models/lato2
  config   lato2_config.json, only when --models-dir moves the weights away from the default

ComfyUI's own Python is left alone. The environment is built with uv (installed with pip if missing).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
ENV_DIR = NODE_DIR / "envs" / "lato2"
REPO_DIR = NODE_DIR / "vendor" / "LATO.2"
CONFIG = NODE_DIR / "lato2_config.json"
DEFAULT_MODELS = NODE_DIR.parents[1] / "models" / "lato2"
LATO2_REPO = "https://github.com/LoHhhha/LATO.2.git"
LATO2_COMMIT = "fbb1f5a5755e6db8700cf6922fd506830b7cdccd"  # 2026-08-15, the code this node was tested with
WEIGHTS_REPO = "0x4c48/LATO.2"
WEIGHTS_REVISION = "a91090e8077b9318ab87ac08fd9eb905903d4515"
TORCH = ["torch==2.8.0", "torchvision==0.23.0", "--index-url", "https://download.pytorch.org/whl/cu128"]
WINDOWS = os.name == "nt"
UV: list[str] = []


def step(text):
    print(f"[lato2] {time.strftime('%H:%M:%S')} {text}", flush=True)


def run(*command, cwd=None):
    step("$ " + " ".join(str(c) for c in command)[:200])
    subprocess.run([str(c) for c in command], cwd=cwd, check=True)


def python_in(env: Path) -> Path:
    return env / ("Scripts/python.exe" if WINDOWS else "bin/python")


def uv_command() -> list[str]:
    """How to run uv: the one installed into this Python, this Python's `-m uv`, a uv program on PATH, or uv
    installed into this Python now. Not a launcher shim on PATH (pyenv-win's uv.bat): it passes the arguments
    through cmd, which reads the '<' in 'numpy<2.3' as an input redirection."""
    own = Path(sysconfig.get_path("scripts")) / ("uv.exe" if WINDOWS else "uv")
    if own.is_file():
        return [str(own)]
    if module_uv():
        return [sys.executable, "-m", "uv"]
    found = shutil.which("uv")
    if found and not (WINDOWS and found.lower().endswith((".bat", ".cmd"))):
        return [found]
    step("installing uv")
    run(sys.executable, "-m", "pip", "install", "--quiet", "uv")
    return [sys.executable, "-m", "uv"]


def module_uv() -> bool:
    try:
        subprocess.run([sys.executable, "-m", "uv", "--version"], check=True, capture_output=True)
        return True
    except Exception:
        return False


def repo():
    if (REPO_DIR / ".git").is_dir():
        step(f"LATO.2 code already in {REPO_DIR}")
        return
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    run("git", "init", "--quiet", cwd=REPO_DIR)
    run("git", "remote", "add", "origin", LATO2_REPO, cwd=REPO_DIR)
    run("git", "fetch", "--quiet", "--depth", "1", "origin", LATO2_COMMIT, cwd=REPO_DIR)
    run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=REPO_DIR)


def env():
    step("Python environment (nothing to compile)")
    run(*UV, "venv", "--allow-existing", "--python", "3.12", ENV_DIR)
    py = python_in(ENV_DIR)
    run(*UV, "pip", "install", "--python", py, *TORCH)
    run(*UV, "pip", "install", "--python", py, "numpy<2.3", "scipy", "trimesh", "pillow", "tqdm", "psutil",
        "easydict", "huggingface_hub", "safetensors", "spconv-cu126==2.3.8")
    try:
        # The build for PyTorch 2.8; --no-deps keeps the CUDA build of PyTorch installed above.
        run(*UV, "pip", "install", "--python", py, "xformers==0.0.32.post2", "--no-deps")
    except subprocess.CalledProcessError:
        step("no xformers wheel for this system (the PyTorch attention stand-in is used instead)")
    run(py, "-c", "import torch, spconv.pytorch, trimesh, scipy\n"
                  "assert torch.cuda.is_available(), 'PyTorch does not see a CUDA GPU'\n"
                  "print('[lato2] environment ready:', torch.__version__, torch.cuda.get_device_name(0))")


def models(models_dir: Path):
    py = python_in(ENV_DIR)
    step(f"LATO.2 weights -> {models_dir}")
    run(py, "-c", "import sys\nfrom huggingface_hub import snapshot_download\n"
                  "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])",
        WEIGHTS_REPO, WEIGHTS_REVISION, models_dir)
    # LATO.2 conditions on DINOv2 through torch.hub: fetch its code and weights now, not on the first run.
    step(f"DINOv2 -> {models_dir / 'dinov2'}")
    run(py, "-c", "import sys, torch\ntorch.hub.set_dir(sys.argv[1])\n"
                  "torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)",
        models_dir / "dinov2")


def config(models_dir: Path):
    data = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.is_file() else {}
    if models_dir.resolve() == DEFAULT_MODELS.resolve():
        data.pop("ckpt", None)
        data.pop("dino_hub_dir", None)
    else:
        data.update(ckpt=str(models_dir), dino_hub_dir=str(models_dir / "dinov2"))
    if data:
        CONFIG.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        step(f"wrote {CONFIG}")
    elif CONFIG.is_file():
        CONFIG.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS,
                        help=f"where the weights go (default {DEFAULT_MODELS})")
    parser.add_argument("--only", default="repo,env,models,config", help="comma-separated steps to run")
    args = parser.parse_args()
    steps = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = steps - {"repo", "env", "models", "config"}
    if unknown:
        raise SystemExit(f"unknown steps: {', '.join(sorted(unknown))}")
    models_dir = args.models_dir.resolve()
    global UV
    UV = uv_command()
    if "repo" in steps:
        repo()
    if "env" in steps:
        env()
    if "models" in steps:
        models(models_dir)
    if "config" in steps:
        config(models_dir)
    step("done: restart ComfyUI, then run LATO.2 Setup Check or LATO.2 Low-Poly Remesh")


if __name__ == "__main__":
    main()
