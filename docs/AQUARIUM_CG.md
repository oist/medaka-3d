# Tracked aquarium CG preview

The diagnostic renderer replays existing C34 slot assignments; it does not run
or modify analysis-by-synthesis tracking. Its three coloured fish are schematic
meshes positioned at the original recorded pectoral anchors. Mouth and tail
positions are reconstructed with the calibrated refractive cameras from the
same stored source rows. Width, fins and eyes are illustrative. There is no
temporal smoothing, interpolation, synthetic tail animation or slot relabelling.

## First preview, 2026-09-16

- Experiment: `20251225_K_1_2_FCMC`, C34 lambda 0.767 (the existing viewing arm).
- Native frames: `75270:78590`, end exclusive, 3,320 frames at 166 fps.
- Main movie: 20 seconds, 600 frames, 1600 x 900, H.264 at 30 fps.
- Source recording time: 453.434 to 473.434 seconds.
- Tank interior: 192 x 192 x 186 mm; z is height above the floor.
- Output: `work/aquarium_cg/c34_75270_78590/`.
- `index.html`: local player, slow playback, event seeking and candidate list.
- `aquarium.mp4`: oblique aquarium view plus simultaneous XY and XZ views.
- `teleport_slow_5.53x.mp4` / `slow.html`: source frames `76100:76432`, all
  332 native frames played at 30 fps (11.067 seconds). The on-frame clock
  continues to show the original two seconds, not the slowed movie duration.
- `native_pose.npz`: all native frames, original anchors/source/slot_track,
  derived pose and residuals, explicit movie sampling indices.
- `metadata.json` and provenance sidecars: paths, hashes, settings and events.

The main window contains 21 adjacent native-frame pectoral steps over 4 mm and
two post-gap reappearances. The largest step is 14.613 mm for slot 0 at f76235
(+5.813 s in the main clip, +4.5 s in the slow clip). This window includes the
previously documented f76930 identity ambiguity. Large displacement is a review
signal, not an identity-swap test: a crossing can swap with little displacement.
The colours name stored correspondence slots, not verified animal identities.

All events are detected before movie downsampling and marked for 0.7 source
seconds. The main video can omit very brief poses; the slow clip retains every
native frame around the largest event. Trails retain all native samples, break
at missing observations and jumps, and do not draw jumps as ordinary swimming.
Missing anchors disappear; missing mouth/tail points show partial geometry.
Out-of-tank estimates are not clamped. Both anchor gaps are single-frame gaps,
so the 30 fps movie may not display the disappearance itself; the reappearance
marker persists, and the full gap remains in the NPZ.

## Reproduce

From the repository root in PowerShell:

```powershell
$inputs = 'work/generative_tracking_inputs'
$baseline = "$inputs/work/c34_microflip/20251225_K_1_2_FCMC__chain-c28-c33-c30v4-c34/C34-label__20251225_K_1_2_FCMC__lam0.767.npz"
$calibration = "$inputs/calibration/fixed_refractive_192x192x186/phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
.venv/Scripts/python.exe -m medaka3d.aquarium `
  --tracks $baseline `
  --detections-dir "$inputs/work/c01_s16_predictions/bottomup" `
  --calibration $calibration `
  --experiment 20251225_K_1_2_FCMC `
  --start-frame 75270 --seconds 20 `
  --output work/aquarium_cg/new_preview
```

Requires the existing project environment (NumPy/OpenCV) and `ffmpeg` on PATH.
Existing nonempty output directories are refused. Set `--seconds` to 10 or 30
and choose another `--start-frame` for another interval. For an all-frame slow
clip, render two source seconds using `--fps 166`, then retime the resulting
video with `ffmpeg -vf "setpts=166/30*PTS" -r 30 -frames:v 332`. The first run's
exact conversion and provenance wrapper is in
`work/aquarium_cg/make_slow_preview.py`.

## Verification

Focused tests cover frame-local source rows, slot order, missing geometry,
native-frame out-and-back jumps that movie downsampling omits, separate
reappearance events, raw trail breaks and movie duration. The stored original
pectoral anchors are retained exactly; independently reconstructed pectoral
positions differ by at most 0.0000124 mm in this first window.
