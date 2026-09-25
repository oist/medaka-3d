"""Synthetic asset bindings and tensor preparation only; no model import/load."""

import copy
import json
import sys

import numpy as np
import pytest
import smoke_dinov2_cpu as m


@pytest.fixture
def asset_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m.assets, "ASSETS", tmp_path)
    monkeypatch.setattr(m.assets, "runtime", lambda: {"synthetic_fixture": True})

    def download(item, root, opener):
        path = root / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# Synthetic metadata fixture; never use for model execution.\n")
        return dict(
            **item, size_bytes=path.stat().st_size, sha256=m.assets.sha(path), published_sha256=None
        )

    monkeypatch.setattr(m.assets, "download", download)
    root = tmp_path / "assets"
    m.assets.acquire(root)
    return root, m.assets.sha(root / "acquisition.provenance.json")


def test_prepared_float32_chw_bytes_and_input_nonmutation():
    chw = np.arange(3 * 224 * 224, dtype=np.float32).reshape(3, 224, 224)[:, :, ::-1]
    before = chw.tobytes()
    batch = m.prepare_input(chw)
    assert batch.shape == (1, 3, 224, 224) and batch.dtype == np.float32
    assert batch.flags.c_contiguous and batch.tobytes() == before
    assert not np.shares_memory(chw, batch) and chw.tobytes() == before


@pytest.mark.parametrize("case", ["rgb_uint8", "float64", "batch", "wrong_size", "nan", "inf"])
def test_invalid_prepared_crop_is_rejected(case):
    values = np.zeros((3, 224, 224), np.float32)
    if case == "rgb_uint8":
        values = np.zeros((224, 224, 3), np.uint8)
    elif case == "float64":
        values = values.astype(np.float64)
    elif case == "batch":
        values = values[None]
    elif case == "wrong_size":
        values = values[:, :223]
    else:
        values[0, 0, 0] = np.nan if case == "nan" else np.inf
    with pytest.raises(ValueError):
        m.prepare_input(values)


def test_asset_binding_verification_does_not_import_model_or_torch(asset_fixture):
    root, seal_sha = asset_fixture
    before = set(sys.modules)
    manifest, evidence = m.verify_assets(root, seal_sha)
    assert manifest["commit"] == m.assets.COMMIT
    assert len(manifest["assets"]) == 20
    assert len(evidence) == 25
    assert not any(name == "dinov2" or name.startswith("dinov2.") for name in sys.modules)
    assert set(sys.modules) - before == set()


@pytest.mark.parametrize("fault", ["weight", "source", "bytecode", "unknown_source", "seal_hash"])
def test_changed_or_extra_asset_is_rejected(asset_fixture, fault):
    root, seal_sha = asset_fixture
    if fault == "weight":
        (root / "weights/dinov2_vits14_pretrain.pth").write_bytes(b"changed")
    elif fault == "source":
        (root / "source/dinov2/hub/backbones.py").write_bytes(b"changed")
    elif fault == "bytecode":
        (root / "source/dinov2/forged.pyc").write_bytes(b"bad bytecode")
    elif fault == "unknown_source":
        (root / "source/dinov2/other.py").write_bytes(b"raise AssertionError")
    else:
        seal_sha = "0" * 64
    with pytest.raises(ValueError):
        m.verify_assets(root, seal_sha)


@pytest.mark.parametrize(
    "field,value",
    [
        ("commit", "0" * 40),
        ("model", "dinov2_vits14_reg"),
        ("with_registers", True),
        ("license", "other"),
        ("status", "failed"),
    ],
)
def test_even_rebound_manifest_cannot_change_fixed_specification(asset_fixture, field, value):
    root, _ = asset_fixture
    manifest_path = root / "assets.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    seal_path = root / "acquisition.provenance.json"
    seal = json.loads(seal_path.read_text())
    item = seal["outputs"]["assets.manifest.json"]
    item["sha256"] = m.assets.sha(manifest_path)
    item["size_bytes"] = manifest_path.stat().st_size
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="specification"):
        m.verify_assets(root, m.assets.sha(seal_path))


def test_asset_input_digest_and_output_metadata_are_not_mutated(asset_fixture):
    root, seal_sha = asset_fixture
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    _, evidence = m.verify_assets(root, seal_sha)
    assert copy.deepcopy(evidence) == evidence
    assert all(p.read_bytes() == body for p, body in before.items())
