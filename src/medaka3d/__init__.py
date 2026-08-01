"""medaka3d — 3-camera 3D behaviour tracking for medaka mating experiments.

A rebuild of `medaka-mating-pipeline`, starting from raw video. See
``docs/HANDOFF.md`` for what was inherited and why, and ``docs/PRIOR_FINDINGS.md``
for prior results — all of which are recorded as hypotheses to re-verify, not as
established facts.
"""

from medaka3d.assets import AssetCatalog, AssetError, load_catalog
from medaka3d.paths import PathConfigError, Paths, load_paths
from medaka3d.provenance import provenance

__all__ = [
    "AssetCatalog",
    "AssetError",
    "PathConfigError",
    "Paths",
    "load_catalog",
    "load_paths",
    "provenance",
]

__version__ = "0.0.0"
