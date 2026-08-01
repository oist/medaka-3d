"""Structural rules that keep experiments from becoming permanent.

These tests are the anti-mess mechanism. The retired repo relied on discipline
to keep experimental code out of the production path and to retire finished
experiments; discipline lost. Here the build fails instead.
"""

from __future__ import annotations

import ast
import datetime as dt
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LABS = REPO / "labs"
PRODUCTION = REPO / "src" / "medaka3d"

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
REQUIRED_KEYS = ("hypothesis", "stage", "expires", "decision_rule")
VALID_VERDICTS = ("promoted", "rejected", "parked")


def lab_dirs() -> list[Path]:
    if not LABS.is_dir():
        return []
    return sorted(p for p in LABS.iterdir() if p.is_dir() and not p.name.startswith("_"))


def parse_front_matter(readme: Path) -> dict[str, str]:
    """Minimal YAML-ish front matter reader: top-level `key:` pairs only."""
    match = FRONT_MATTER.search(readme.read_text(encoding="utf-8"))
    if not match:
        return {}
    fields: dict[str, str] = {}
    key = None
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if not line[0].isspace() and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            fields[key] = value.strip().lstrip(">|").strip()
        elif key:
            fields[key] = (fields[key] + " " + line.strip()).strip()
    return fields


def imported_roots(source: str) -> set[str]:
    """Top-level package names imported by a module."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class TestIsolation:
    def test_production_does_not_import_labs(self):
        """The wall. Experimental code must never reach the production line."""
        offenders = [
            path.relative_to(REPO).as_posix()
            for path in PRODUCTION.rglob("*.py")
            if "labs" in imported_roots(path.read_text(encoding="utf-8"))
        ]
        assert not offenders, (
            "production code imports from labs/: "
            + ", ".join(offenders)
            + "\nPromote the component into src/medaka3d/ instead."
        )

    def test_labs_do_not_import_each_other(self):
        """Labs are independent. Cross-imports resurrect the old tangle."""
        names = {d.name for d in lab_dirs()}
        offenders: list[str] = []
        for lab in lab_dirs():
            for path in lab.rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                for other in imported_roots(source) & names:
                    if other != lab.name:
                        offenders.append(f"{path.relative_to(REPO).as_posix()} -> {other}")
        assert not offenders, "labs import each other: " + ", ".join(offenders)


class TestLabHygiene:
    def test_template_is_present(self):
        assert (LABS / "_template" / "README.md").is_file(), "labs/_template/README.md is missing"

    @pytest.mark.parametrize("lab", lab_dirs(), ids=lambda p: p.name)
    def test_lab_is_well_formed(self, lab: Path):
        assert re.fullmatch(r"\d{8}_[a-z0-9_]+", lab.name), (
            f"{lab.name}: directory must be named YYYYMMDD_slug (lowercase)"
        )

        readme = lab / "README.md"
        assert readme.is_file(), f"{lab.name}: README.md is required"

        fields = parse_front_matter(readme)
        missing = [k for k in REQUIRED_KEYS if not fields.get(k)]
        assert not missing, f"{lab.name}: front matter missing {', '.join(missing)}"

        from medaka3d.stages import PROTOCOLS

        assert fields["stage"] in PROTOCOLS, (
            f"{lab.name}: stage {fields['stage']!r} is not one of {sorted(PROTOCOLS)}"
        )
        try:
            expires = dt.date.fromisoformat(fields["expires"])
        except ValueError:
            pytest.fail(f"{lab.name}: expires must be ISO YYYY-MM-DD, got {fields['expires']!r}")

        result = lab / "RESULT.md"
        if dt.date.today() > expires and not result.is_file():
            pytest.fail(
                f"{lab.name} expired on {expires} with no RESULT.md.\n"
                f"Write the verdict (promoted / rejected / parked) or delete the "
                f"directory. Leaving it is how the previous repository accumulated "
                f"53 experiments that nobody could safely remove."
            )

        if result.is_file():
            text = result.read_text(encoding="utf-8").lower()
            assert any(v in text for v in VALID_VERDICTS), (
                f"{lab.name}/RESULT.md must state a verdict: {', '.join(VALID_VERDICTS)}"
            )
