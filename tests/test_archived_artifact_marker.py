"""Optional archive checks must still fail when present evidence is invalid."""

import subprocess
import sys
from pathlib import Path


def test_missing_archive_skips_but_present_invalid_evidence_fails(tmp_path):
    (tmp_path / "valid.txt").write_text("sealed", encoding="utf-8")
    (tmp_path / "invalid.txt").write_text("corrupted", encoding="utf-8")
    test_file = tmp_path / "test_original_evidence.py"
    test_file.write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "ROOT = Path(__file__).parent\n"
        "@pytest.mark.requires_artifacts(ROOT / 'absent.txt')\n"
        "def test_absent():\n"
        "    raise AssertionError('missing archive was opened')\n"
        "@pytest.mark.requires_artifacts(ROOT / 'valid.txt')\n"
        "def test_valid():\n"
        "    assert (ROOT / 'valid.txt').read_text() == 'sealed'\n"
        "@pytest.mark.requires_artifacts(ROOT / 'invalid.txt')\n"
        "def test_invalid():\n"
        "    assert (ROOT / 'invalid.txt').read_text() == 'sealed'\n",
        encoding="utf-8",
    )
    # Load the actual repository hooks in a separate process, with no outer
    # pytest state. An existing but invalid archive must never become a skip.
    code = (
        "import sys, pytest; "
        f"sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "import conftest; "
        "raise SystemExit(pytest.main(sys.argv[1:], plugins=[conftest]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(test_file),
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--tb=short",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 passed, 1 skipped" in result.stdout, result.stdout + result.stderr
