"""Source memberships follow each producer's consumed computation."""

import ast
from pathlib import Path

import pytest
import score_bounded_minute_pixels_v2 as score


def fixture(tmp_path):
    names = [str((tmp_path / (name + ".py")).resolve()) for name in ("renderer", "motion")]
    hashes = dict.fromkeys(names, "a" * 64)
    prep = dict(inputs={p: dict(sha256=v) for p, v in hashes.items()})
    old = dict(inputs={names[0]: dict(sha256=hashes[names[0]])})
    return names, hashes, prep, old


def test_new_unused_import_does_not_require_false_old_provenance(tmp_path):
    names, hashes, prep, old = fixture(tmp_path)
    assert score.shared_source_bindings(hashes, prep, old, names[:1], hashes) == hashes


@pytest.mark.parametrize("producer", ["prep", "old"])
def test_required_render_source_must_be_bound_by_both_producers(tmp_path, producer):
    names, hashes, prep, old = fixture(tmp_path)
    {"prep": prep, "old": old}[producer]["inputs"].pop(names[0])
    with pytest.raises(ValueError):
        score.shared_source_bindings(hashes, prep, old, names[:1], hashes)


@pytest.mark.parametrize("changed", ["required_old", "required_prep", "extra_prep", "extra_old"])
def test_all_actual_shared_hashes_are_still_checked(tmp_path, changed):
    names, hashes, prep, old = fixture(tmp_path)
    path = names[1] if changed.startswith("extra") else names[0]
    seal = old if changed.endswith("old") else prep
    seal["inputs"][path] = dict(sha256="b" * 64)
    with pytest.raises(ValueError, match="sealed input differs"):
        score.shared_source_bindings(hashes, prep, old, names[:1], hashes)


def test_numerical_assay_and_frozen_sampling_scope_are_unchanged():
    trees = []
    for filename in ("score_bounded_minute_pixels.py", "score_bounded_minute_pixels_v2.py"):
        tree = ast.parse(Path(score.__file__).with_name(filename).read_text(encoding="utf-8"))
        trees.append(
            {
                f.name: ast.dump(f, include_attributes=False)
                for f in tree.body
                if isinstance(f, ast.FunctionDef)
            }
        )
    for name in ("squared_error", "validate_saved_frame", "summary_errors", "bound_input"):
        assert trees[0][name] == trees[1][name]
    assert score.prior.SCORED.tolist() == [66483 + 166 * k for k in range(60)]
