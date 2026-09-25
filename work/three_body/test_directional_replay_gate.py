"""A waiting branch must not proceed on a missing or unsuccessful owned exit."""

import json
from types import SimpleNamespace

import pytest
import replay_directional_chest as branch


class Window:
    def __init__(self):
        self.calls = 0

    def assert_open(self):
        self.calls += 1


def test_missing_owned_exit_times_out_without_reading_qualification(tmp_path, monkeypatch):
    times = iter((0.0, 901.0))
    monkeypatch.setattr(branch.time, "monotonic", lambda: next(times))
    window = Window()
    evidence = SimpleNamespace(add=lambda *a: pytest.fail("must not accept any exit evidence"))
    with pytest.raises(ValueError, match="exit witness timeout"):
        branch.await_qualification_exit(tmp_path, tmp_path / "absent.json", window, evidence)
    assert window.calls == 1


@pytest.mark.parametrize("session,exit_code", [(96265, 1), (96264, 0)])
def test_wrong_or_failed_owned_exit_cannot_release_branch(tmp_path, session, exit_code):
    witness = tmp_path / "witness.json"
    witness.write_text(
        json.dumps(
            dict(
                schema="ROOT_owned_directional_CUDA_exit_v1",
                exec_session_id=session,
                exit_code=exit_code,
                qualification_root=str(tmp_path),
            )
        ),
        encoding="utf-8",
    )
    evidence = SimpleNamespace(add=lambda *a: pytest.fail("must not accept failed exit evidence"))
    with pytest.raises(ValueError, match="successful ROOT-owned"):
        branch.await_qualification_exit(tmp_path, witness, Window(), evidence)
