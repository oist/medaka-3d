"""Resolution and integrity-checking of inherited assets.

Inherited models and training data live outside the repo, under
``paths.asset_root``. A ``MANIFEST.json`` at that root is the contract: it names
every asset, its size, its sha256, and where it came from.

Why this exists: in the old repo the answer to "which model produced this
result" lived in prose across `AGENTS.md`, `MEMORY.md` and a YAML that pointed
at deleted files, and the three disagreed. Here it is one machine-readable file
that is verified before use.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 20


class AssetError(RuntimeError):
    """Raised when an asset is missing, altered, or not in the manifest."""


@dataclass(frozen=True, slots=True)
class Asset:
    """One inherited file, as declared by the manifest."""

    key: str
    path: Path
    size: int
    sha256: str
    kind: str
    provenance: str

    def verify(self, *, deep: bool = False) -> None:
        """Check the asset on disk against the manifest.

        Args:
            deep: If True, re-hash the file. Slow for multi-GB checkpoints, so
                the default is a size check only. Use ``deep=True`` after any
                copy between machines or filesystems.
        """
        if not self.path.is_file():
            raise AssetError(f"{self.key}: missing at {self.path}")
        actual_size = self.path.stat().st_size
        if actual_size != self.size:
            raise AssetError(
                f"{self.key}: size mismatch at {self.path} "
                f"(manifest {self.size}, on disk {actual_size})"
            )
        if deep:
            actual = sha256_file(self.path)
            if actual != self.sha256:
                raise AssetError(
                    f"{self.key}: sha256 mismatch at {self.path}\n"
                    f"  manifest {self.sha256}\n  on disk  {actual}"
                )


class AssetCatalog:
    """The parsed ``MANIFEST.json``, keyed by asset name."""

    def __init__(self, root: Path, assets: dict[str, Asset], meta: dict) -> None:
        self.root = root
        self.meta = meta
        self._assets = assets

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self) -> Iterator[Asset]:
        return iter(self._assets.values())

    def __contains__(self, key: object) -> bool:
        return key in self._assets

    def get(self, key: str, *, verify: bool = True, deep: bool = False) -> Path:
        """Resolve an asset key to a verified path on disk."""
        try:
            asset = self._assets[key]
        except KeyError:
            raise AssetError(
                f"Unknown asset {key!r}. Known keys: {', '.join(sorted(self._assets))}"
            ) from None
        if verify:
            asset.verify(deep=deep)
        return asset.path

    def of_kind(self, kind: str) -> list[Asset]:
        """All assets of one kind, e.g. ``sleap_model`` or ``training_labels``."""
        return [a for a in self._assets.values() if a.kind == kind]

    def verify_all(self, *, deep: bool = False) -> list[str]:
        """Verify every asset; return the list of problems rather than raising."""
        problems: list[str] = []
        for asset in self._assets.values():
            try:
                asset.verify(deep=deep)
            except AssetError as exc:
                problems.append(str(exc))
        return problems


def load_catalog(asset_root: Path) -> AssetCatalog:
    """Read and parse ``MANIFEST.json`` from the asset root."""
    manifest_path = Path(asset_root) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise AssetError(
            f"No {MANIFEST_NAME} at {asset_root}. The asset root must carry a "
            f"manifest; see docs/ASSET_MANIFEST.md."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assets: dict[str, Asset] = {}
    for key, entry in payload.get("assets", {}).items():
        assets[key] = Asset(
            key=key,
            path=Path(asset_root) / entry["path"],
            size=int(entry["size"]),
            sha256=entry["sha256"],
            kind=entry.get("kind", "unknown"),
            provenance=entry.get("provenance", ""),
        )
    meta = {k: v for k, v in payload.items() if k != "assets"}
    return AssetCatalog(root=Path(asset_root), assets=assets, meta=meta)


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file, safe for multi-GB checkpoints."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
