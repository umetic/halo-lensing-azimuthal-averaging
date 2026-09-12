from __future__ import annotations

import json
from pathlib import Path

import pytest

from azlens.manifest import (
    collect_git_state,
    file_manifest,
    json_dumps_stable,
    package_versions,
    require_clean_git,
    sha256_file,
    write_json,
)


def test_sha256_file_and_file_manifest(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    records = file_manifest([path], base=tmp_path)
    assert records == [{"path": "sample.txt", "bytes": 3, "sha256": sha256_file(path)}]


def test_json_writer_is_stable(tmp_path: Path):
    path = tmp_path / "nested" / "data.json"
    write_json(path, {"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == json_dumps_stable({"a": 1, "b": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


def test_package_versions_reports_missing_package():
    versions = package_versions(["numpy", "definitely_not_an_installed_azlens_dependency"])
    assert versions["numpy"] != "not installed"
    assert versions["definitely_not_an_installed_azlens_dependency"] == "not installed"


def test_git_state_non_repository(tmp_path: Path):
    state = collect_git_state(tmp_path)
    assert not state.available
    assert state.dirty is None
    assert require_clean_git(tmp_path).available is False


def test_require_clean_git_rejects_dirty_tree_when_git_available(tmp_path: Path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git executable not available")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (tmp_path / "tracked.txt").write_text("two\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        require_clean_git(tmp_path)
    assert require_clean_git(tmp_path, allow_dirty=True).dirty is True
