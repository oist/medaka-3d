"""Tests for the three foundation modules: paths, assets, provenance."""

from __future__ import annotations

import json

import pytest

from medaka3d.assets import AssetError, load_catalog, sha256_file
from medaka3d.paths import PathConfigError, load_paths
from medaka3d.provenance import provenance


def write_config(tmp_path, **overrides):
    """Write a paths.local.toml-shaped file, with real directories."""
    values = {
        "raw_data_dir": tmp_path / "raw",
        "asset_root": tmp_path / "assets",
        "work_dir": tmp_path / "work",
    }
    values.update(overrides)
    for key, directory in values.items():
        if key != "work_dir":  # work_dir is created by the loader
            directory.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'{k} = "{str(v).replace(chr(92), chr(92) * 2)}"' for k, v in values.items())
    config = tmp_path / "paths.local.toml"
    config.write_text(f"[paths]\n{body}\n", encoding="utf-8")
    return config


class TestPaths:
    def test_optional_appearance_staging_is_checked_without_creation(self, tmp_path):
        staging = tmp_path / "appearance"
        config = write_config(tmp_path, appearance_staging_dir=staging)
        assert load_paths(config).appearance_staging_dir == staging
        staging.rmdir()
        with pytest.raises(PathConfigError, match="not exist"):
            load_paths(config)
        assert not staging.exists()

    def test_loads_and_creates_work_dir(self, tmp_path):
        config = write_config(tmp_path)
        paths = load_paths(config)
        assert paths.raw_data_dir == tmp_path / "raw"
        assert paths.work_dir.is_dir(), "work_dir is the one root we may create"

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(PathConfigError, match="No path config"):
            load_paths(tmp_path / "absent.toml")

    def test_missing_required_key_is_an_error(self, tmp_path):
        config = tmp_path / "p.toml"
        config.write_text('[paths]\nraw_data_dir = "x"\n', encoding="utf-8")
        with pytest.raises(PathConfigError, match="missing required key"):
            load_paths(config)

    def test_nonexistent_directory_is_an_error(self, tmp_path):
        """The failure the old repo never had: a configured path that is gone."""
        config = write_config(tmp_path)
        (tmp_path / "raw").rmdir()
        with pytest.raises(PathConfigError, match="not exist"):
            load_paths(config)

    def test_unknown_key_is_an_error(self, tmp_path):
        config = write_config(tmp_path)
        config.write_text(
            config.read_text(encoding="utf-8") + 'sleap_models_dir = "stale"\n',
            encoding="utf-8",
        )
        with pytest.raises(PathConfigError, match="unrecognised key"):
            load_paths(config)

    def test_video_requires_exactly_one_match(self, tmp_path):
        config = write_config(tmp_path)
        paths = load_paths(config)
        view = tmp_path / "raw" / "EXP_1" / "top_cam"
        view.mkdir(parents=True)

        with pytest.raises(PathConfigError, match="found 0"):
            paths.video("EXP_1", "top_cam")

        (view / "cam0_top.mp4").touch()
        assert paths.video("EXP_1", "top_cam").name == "cam0_top.mp4"

        (view / "cam0_top_RETAKE.mp4").touch()
        with pytest.raises(PathConfigError, match="found 2"):
            paths.video("EXP_1", "top_cam")

    def test_unknown_experiment_is_an_error(self, tmp_path):
        paths = load_paths(write_config(tmp_path))
        with pytest.raises(PathConfigError, match="No such experiment"):
            paths.experiment_dir("NOPE")


class TestAssets:
    def build_root(self, tmp_path, payload=b"weights"):
        root = tmp_path / "assets"
        root.mkdir(exist_ok=True)
        model = root / "models" / "m.ckpt"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(payload)
        manifest = {
            "created": "2026-08-01",
            "assets": {
                "sleap.round8.top.centroid": {
                    "path": "models/m.ckpt",
                    "size": len(payload),
                    "sha256": sha256_file(model),
                    "kind": "sleap_model",
                    "provenance": "retrain_20260722_round8",
                }
            },
        }
        (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root, model

    def test_resolves_and_verifies(self, tmp_path):
        root, model = self.build_root(tmp_path)
        catalog = load_catalog(root)
        assert len(catalog) == 1
        assert catalog.get("sleap.round8.top.centroid", deep=True) == model
        assert catalog.of_kind("sleap_model")[0].provenance == "retrain_20260722_round8"

    def test_absolute_path_references_canonical_location(self, tmp_path):
        """Large checkpoints stay where they are; the manifest points at them."""
        root, _ = self.build_root(tmp_path)
        elsewhere = tmp_path / "canonical" / "big.ckpt"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_bytes(b"elsewhere")
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["assets"]["sleap.round7.side.centroid"] = {
            "path": str(elsewhere),
            "size": elsewhere.stat().st_size,
            "sha256": sha256_file(elsewhere),
            "kind": "sleap_model",
            "provenance": "retrain_20260704",
        }
        (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

        catalog = load_catalog(root)
        assert catalog.get("sleap.round7.side.centroid", deep=True) == elsewhere
        assert catalog.verify_all(deep=True) == []

    def test_missing_manifest_is_an_error(self, tmp_path):
        (tmp_path / "bare").mkdir()
        with pytest.raises(AssetError, match=r"No MANIFEST\.json"):
            load_catalog(tmp_path / "bare")

    def test_unknown_key_lists_what_is_available(self, tmp_path):
        root, _ = self.build_root(tmp_path)
        with pytest.raises(AssetError, match=r"sleap\.round8\.top\.centroid"):
            load_catalog(root).get("sleap.round9.top.centroid")

    def test_detects_a_deleted_asset(self, tmp_path):
        root, model = self.build_root(tmp_path)
        catalog = load_catalog(root)
        model.unlink()
        with pytest.raises(AssetError, match="missing"):
            catalog.get("sleap.round8.top.centroid")

    def test_detects_a_corrupted_asset(self, tmp_path):
        root, model = self.build_root(tmp_path)
        catalog = load_catalog(root)
        model.write_bytes(b"WEIGHTS")  # same length as b"weights", different bytes
        catalog.get("sleap.round8.top.centroid", deep=False)  # size check passes
        with pytest.raises(AssetError, match="sha256 mismatch"):
            catalog.get("sleap.round8.top.centroid", deep=True)

    def test_verify_all_collects_rather_than_raises(self, tmp_path):
        root, model = self.build_root(tmp_path)
        catalog = load_catalog(root)
        model.unlink()
        assert len(catalog.verify_all()) == 1


class TestProvenance:
    def read_sidecar(self, out_dir, stage):
        return json.loads((out_dir / f"{stage}.provenance.json").read_text(encoding="utf-8"))

    def test_records_inputs_outputs_params(self, tmp_path):
        source = tmp_path / "in.slp"
        source.write_bytes(b"x" * 10)
        out = tmp_path / "out"

        with provenance("reconstruct_3d", out) as record:
            record.input("top_slp", source)
            record.param("max_jump_mm", 8)
            record.output("tracks", tmp_path / "tracks.csv")

        data = self.read_sidecar(out, "reconstruct_3d")
        assert data["status"] == "ok"
        assert data["inputs"]["top_slp"]["size"] == 10
        assert data["params"]["max_jump_mm"] == 8
        assert data["outputs"]["tracks"]["missing"] is True
        assert "argv" in data["environment"]

    def test_records_failure_and_reraises(self, tmp_path):
        """A crashed run you cannot diagnose is worse than no run at all."""
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="boom"), provenance("stage", out):
            raise ValueError("boom")

        data = self.read_sidecar(out, "stage")
        assert data["status"] == "failed"
        assert "boom" in data["error"]
