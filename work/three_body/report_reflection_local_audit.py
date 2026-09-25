"""Seal measured interpretation separately from the unchanged numerical audit."""

import hashlib
import json
import shutil
from pathlib import Path

from medaka3d.provenance import provenance

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "work/three_body/reflection_local_probe_independent_audit_v1"
OUTPUT = ROOT / "interpretation_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    seal_path = ROOT / "audit.provenance.json"
    audit = json.loads(seal_path.read_text())
    if audit["status"] != "ok":
        raise ValueError("successful numerical audit required")
    inputs = {str(seal_path): sha(seal_path), str(Path(__file__).resolve()): sha(__file__)}
    for entry in audit["outputs"].values():
        inputs[entry["path"]] = entry["sha256"]
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError("audit output changed")
    OUTPUT.mkdir(exist_ok=False)
    with provenance("reflection_local_audit_interpretation", OUTPUT, name="report") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        source = OUTPUT / Path(__file__).name
        shutil.copyfile(__file__, source)
        text = """# Reflection-aware local fitting: measured interpretation

Both reflected arms pass the registered development gates. The independent audit
recomputed the complete 3×3 scoring matrix from saved pixel responses, all 39
final strict ray/projection supports, fixed plans/priors, and 585 stored iterate
records. It verified all 65 producer outputs and the earlier registration.
No pixel rendering, fitting, raw video decoding or GT access was repeated.

| Fitted pose | Direct scorer | Unit scorer | Fresnel scorer |
|---|---:|---:|---:|
| Direct | 0.445106253 | 0.424360424 | 0.432096443 |
| Unit | 0.446210112 | 0.418523118 | 0.431535828 |
| Fresnel | 0.445029472 | 0.422688401 | 0.430848649 |

Holding the scoring model fixed, the unit fit lowers mean error by 0.005837306
and the Fresnel fit by 0.001247794, compared with the refitted direct pose.
These changes exceed the registered 1e-10 requirement. Under the common
direct-only scorer, unit worsens by 0.001103860 while Fresnel improves by only
0.000076780. Much of the larger diagonal improvement comes from the changed
image-response model; it must not all be attributed to improved geometry.

There is **no support recovery** in this local comparison: all three arms have
exactly the same observed mask at every scene/slot/node, with 27 mouth, 25
pectoral and 23 tail supports out of 39 opportunities each. Thus there are no
support gains or losses, including 79372, 82719 and 82720. This makes the
no-support-loss gates pass but does not demonstrate recapture or identity repair.

| Same-model pose change | top | side1 | side2 |
|---|---:|---:|---:|
| Unit | -0.005591505 | -0.014523004 | +0.002602591 |
| Fresnel | -0.004482026 | -0.000110908 | +0.000849553 |

Negative means lower error. Both average gains coexist with a worse side2 view.
Across views, unit's foreground/near/global changes are
-0.004358378/-0.001485472/+0.000006543; Fresnel's are
-0.001134084/-0.000113353/-0.000000356. D2-negative pixels are the detector's
negative labels, not independently established physical background.

All 13 scenes remain in the tables and CSV. Unit worsens at 82720
(+0.000021455); its tiny 82719 gain is -1.44e-8. Fresnel worsens at 79316
(+0.000480774), 79322 (+0.000157433), 79372 (+0.000005803), and 82720
(+2.58e-8); 82719 is effectively unchanged at -3.19e-12. These are same-model
pose changes, not comparisons between unlike scoring definitions.

At the motivating 79381 scene, unit's same-model mean change is -0.004999124,
driven by top: top/side1/side2 change -0.020090563/+0.001659078/+0.003434113.
Fresnel changes -0.000970667 overall, with corresponding view changes
-0.002843194/+0.001381599/-0.001450408. Neither creates another direct node
support. The largest node displacement from the direct fit across all scenes
is 2.518117 mm for unit and 2.146187 mm for Fresnel, both at 79396; agreement
of a detector mask objective alone cannot certify these coordinate changes.

The matrix/heatmap PNG was visually checked: frame and camera labels are
readable, blue/red represent lower/worse same-model error, and the image makes
no raw biological ID assertion. The two heatmaps have separate labeled color
scales, so compare their numeric colorbars rather than color saturation alone.

This is an opened-development, fixed-assignment, 14-step local experiment.
The renderer keeps an inherited unbounded direct ray plus nominal finite wall
reflections; the absolute floor survey remains unresolved. Fresnel-weighted
Gaussian alpha is not calibrated radiometry. Temporal branches, flow history,
mask ownership, accepted coverage and animal correspondence were not tested.
A separately registered temporal experiment is justified as a diagnostic next
step; neither reflected arm is adopted by this result.

Numerical detail: [full audit](../RESULT.md), [all view/stratum rows](../all_view_errors.csv),
[cross-evaluation figure](../cross_evaluation.png), [machine-readable audit](../summary.json).
Large video/D2 input hashes were inherited from the producer's successful
verification rather than read again. Live code and every input actually used
by the independent audit were checked before/after, with separate source copies.
"""
        (OUTPUT / "RESULT.md").write_text(text, encoding="utf-8")
        for path in OUTPUT.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=sha(path))
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise ValueError("input changed while interpreting audit")
        for entry in record.outputs.values():
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("report changed")


if __name__ == "__main__":
    main()
