"""Where LATO.2's environment, code and weights live, and how the runner is started.

LATO.2 needs pinned packages that ComfyUI's own Python should not carry, so it runs as a subprocess in its own
environment (install_lato2.py builds it). Paths default to what the installer sets up; lato2_config.json next to
this file overrides them, and LATO2_<KEY> environment variables (LATO2_PYTHON, LATO2_REPO, LATO2_CKPT,
LATO2_DINO_HUB_DIR, LATO2_TIMEOUT) override single keys.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent
RUNNER = NODE_DIR / "runners" / "lato2_runner.py"
CONFIG_PATH = NODE_DIR / "lato2_config.json"
REQUIRED = ("python", "repo", "ckpt")
PROGRESS_PREFIX = "[lato2 progress]"
# On Windows a runner started from a ComfyUI without a console would otherwise open a console window.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def models_dir() -> Path:
    """ComfyUI/models/lato2 (or the same place relative to this folder outside ComfyUI)."""
    try:
        import folder_paths  # type: ignore
        return Path(folder_paths.models_dir) / "lato2"
    except Exception:
        return NODE_DIR.parents[1] / "models" / "lato2"


def env_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def config() -> dict:
    """The installer's layout, then lato2_config.json, then LATO2_* environment variables."""
    ckpt = models_dir()
    data = {"python": str(env_python(NODE_DIR / "envs" / "lato2")), "repo": str(NODE_DIR / "vendor" / "LATO.2"),
            "ckpt": str(ckpt), "dino_hub_dir": str(ckpt / "dinov2"), "timeout": 1800}
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data.update(json.load(handle))
    for key, value in os.environ.items():
        if key.startswith("LATO2_") and key != "LATO2_PORTABLE":
            data[key[len("LATO2_"):].lower()] = value
    return data


def problems() -> list[str]:
    settings = config()
    found = []
    for key in REQUIRED:
        value = settings.get(key)
        if not value:
            found.append(f"{key} is not set")
        elif not Path(value).exists():
            found.append(f"{key} does not exist: {value}")
    if not found and not any(Path(settings["ckpt"]).glob("*.pt")):
        found.append(f"no LATO.2 weights (*.pt) in {settings['ckpt']}")
    return found


def status_text() -> str:
    found = problems()
    settings = config()
    lines = [f"LATO.2: {'ready' if not found else 'not set up'}"]
    lines += [f"  {key}: {settings.get(key)}" for key in ("python", "repo", "ckpt", "dino_hub_dir")]
    if found:
        lines += [f"  problem: {p}" for p in found]
        lines.append(f"  run: python \"{NODE_DIR / 'install_lato2.py'}\"  (then restart ComfyUI)")
    return "\n".join(lines)


def _echo(text: str) -> None:
    """Show a runner's line in ComfyUI's console. A console that cannot encode a character (Windows' cp1252 meets
    tqdm's block characters) gets a stand-in for it rather than failing the node."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


def run(args: list[str], *, on_progress=None) -> str:
    """Run the LATO.2 runner with its environment's Python. Lines starting with "[lato2 progress] N stage" call
    on_progress(N, stage). Returns the combined output; raises RuntimeError with its tail on failure."""
    found = problems()
    if found:
        raise RuntimeError("LATO.2 is not set up: " + "; ".join(found)
                           + f". Run python \"{NODE_DIR / 'install_lato2.py'}\" and restart ComfyUI.")
    settings = config()
    timeout = float(settings.get("timeout") or 1800)
    command = [settings["python"], "-u", str(RUNNER), "--repo", settings["repo"], "--ckpt", settings["ckpt"],
               "--dino_hub_dir", settings.get("dino_hub_dir") or str(Path(settings["ckpt"]) / "dinov2"), *args]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    process = subprocess.Popen(command, cwd=str(NODE_DIR), env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                               creationflags=_NO_WINDOW)
    lines: list[str] = []
    deadline = time.time() + timeout

    def watchdog():
        while process.poll() is None:
            if time.time() > deadline:
                process.kill()
                return
            time.sleep(2)

    threading.Thread(target=watchdog, daemon=True).start()
    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            _echo(f"[LATO.2] {line}")
            if line.startswith(PROGRESS_PREFIX) and on_progress is not None:
                rest = line[len(PROGRESS_PREFIX):].strip().split(" ", 1)
                try:
                    on_progress(float(rest[0]), rest[1] if len(rest) > 1 else "")
                except ValueError:
                    pass
    except BaseException:
        process.kill()  # the node is failing (or was interrupted): do not leave the runner on the GPU
        raise
    code = process.wait()
    output = "\n".join(lines)
    if code != 0:
        tail = "\n".join(lines[-25:])
        if time.time() > deadline:
            raise RuntimeError(f"LATO.2 timed out after {timeout:.0f} s\n{tail}")
        raise RuntimeError(f"LATO.2 failed with exit code {code}\n{tail}")
    return output
