"""install_lato2.py's uv lookup: the uv of the Python running it first, never a launcher shim such as pyenv-win's
uv.bat (its arguments pass through cmd, which read the '<' in 'numpy<2.3' as an input redirection)."""
import sys

import install_lato2


def test_uses_the_uv_installed_into_this_python(tmp_path, monkeypatch):
    uv = tmp_path / ("uv.exe" if install_lato2.WINDOWS else "uv")
    uv.write_bytes(b"")
    monkeypatch.setattr(install_lato2.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(install_lato2.shutil, "which", lambda name: r"C:\pyenv\shims\uv.BAT")
    assert install_lato2.uv_command() == [str(uv)]


def test_skips_a_batch_file_shim_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(install_lato2.sysconfig, "get_path", lambda name: str(tmp_path))     # no uv of its own
    monkeypatch.setattr(install_lato2, "module_uv", lambda: False)
    monkeypatch.setattr(install_lato2, "WINDOWS", True)
    monkeypatch.setattr(install_lato2.shutil, "which", lambda name: r"C:\pyenv\shims\uv.BAT")
    installed = []
    monkeypatch.setattr(install_lato2, "run", lambda *command, cwd=None: installed.append(command))
    assert install_lato2.uv_command() == [sys.executable, "-m", "uv"]
    assert installed and installed[0][-1] == "uv"           # installed into this Python instead


def test_a_real_uv_program_on_path_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(install_lato2.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(install_lato2, "module_uv", lambda: False)
    monkeypatch.setattr(install_lato2.shutil, "which", lambda name: r"C:\tools\uv\uv.exe")
    assert install_lato2.uv_command() == [r"C:\tools\uv\uv.exe"]
