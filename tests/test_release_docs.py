from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_readme_states_scope_and_validation_tag() -> None:
    text = read_text("README.md")
    normalized = text.replace("*", "").lower()
    assert "paper-specific reference implementation" in normalized
    assert "not a general-purpose weak-lensing analysis package" in normalized
    assert "validation-pass-20260913" in text
    assert "Production validation: PASS" in text
    assert "docs/IMPLEMENTATION_HISTORY.md" in text
    assert ("release" + "-candidate") not in normalized


def test_environment_yml_is_named_azlens_and_pins_core_packages() -> None:
    data = yaml.safe_load(read_text("environment.yml"))
    assert data["name"] == "azlens"
    deps = data["dependencies"]
    dep_text = "\n".join(str(item) for item in deps)
    for expected in [
        "python=3.11",
        "numpy=1.26.4",
        "scipy=1.17.1",
        "pandas=3.0.5",
        "matplotlib=3.11.0",
        "pyyaml=6.0.3",
        "pytest=9.0.3",
    ]:
        assert expected in dep_text
    pip_block = next(item for item in deps if isinstance(item, dict) and "pip" in item)
    assert "colossus==1.4.0" in pip_block["pip"]


def test_requirements_matches_public_dependency_set() -> None:
    text = read_text("requirements.txt")
    for expected in [
        "numpy==1.26.4",
        "scipy==1.17.1",
        "pandas==3.0.5",
        "matplotlib==3.11.0",
        "PyYAML==6.0.3",
        "pytest==9.0.3",
        "colossus==1.4.0",
    ]:
        assert expected in text


def test_citation_file_contains_paper_identifier_and_repository() -> None:
    data = yaml.safe_load(read_text("CITATION.cff"))
    assert data["cff-version"] == "1.2.0"
    assert data["title"] == "Halo-Lensing Azimuthal Averaging"
    assert data["repository-code"] == "https://github.com/umetic/halo-lensing-azimuthal-averaging"
    assert data["preferred-citation"]["title"].startswith("Why Azimuthal Averaging Works")
    identifiers = data["preferred-citation"].get("identifiers", [])
    assert any(item.get("value") == "https://arxiv.org/abs/2609.08825" for item in identifiers)


def test_license_is_mit() -> None:
    text = read_text("LICENSE")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Keiichi Umetsu" in text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in text


def test_required_release_documents_exist() -> None:
    for relative in [
        "docs/VALIDATION_SUMMARY.md",
        "docs/REPRODUCTION_WORKFLOW.md",
        "docs/OUTPUT_SCHEMA.md",
        "docs/IMPLEMENTATION_HISTORY.md",
        "docs/RELEASE_NOTES_v0.1.md",
        "docs/RELEASE_CHECKLIST.md",
    ]:
        assert (ROOT / relative).is_file()


def test_development_notes_removed_from_public_docs() -> None:
    assert not list((ROOT / "docs").glob("*" + "incre" + "ment" + "_v0.1.md"))
    docs_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "docs").glob("*.md"))
    lowered = docs_text.lower()
    assert ("pending" + " before public github release") not in lowered
    assert ("release" + " candidate") not in lowered
    assert ("author" + "'" + "s laptop") not in lowered
    assert "no public license is assigned" not in lowered


def test_public_docs_do_not_contain_machine_specific_paths() -> None:
    checked = ["README.md", "CITATION.cff", "environment.yml", "requirements.txt"]
    checked += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "docs").glob("*.md"))]
    checked += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "reference").glob("*/README.md"))]
    # Construct machine-specific path fragments without embedding the full strings directly.
    forbidden = [
        "/" + "home" + "/" + ("kei" + "ichi"),
        ("Drop" + "box") + "/" + "umetic-pc",
        ("mini" + "conda3") + "/" + "envs" + "/" + ("py" + "311"),
    ]
    for relative in checked:
        text = read_text(relative)
        for pattern in forbidden:
            assert pattern not in text, f"{pattern!r} found in {relative}"


def test_validation_summary_records_provenance_without_large_outputs() -> None:
    text = read_text("docs/VALIDATION_SUMMARY.md")
    normalized = text.replace("*", "")
    assert "Product-generation commit: `83b011d126592bffe6dd74cb5f9241e5787dfabf`" in normalized
    assert "Comparison-oracle commit: `161b8b2600631b08a07aa1b5128e45486afba43a`" in normalized
    assert "are not committed to git" in normalized.lower()
    assert "documentation-only changes" in normalized.lower()
