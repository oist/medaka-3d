# Preparing independent image inputs for about 30 minutes

2026-09-09: the input-preparation conditions were registered in `long_input_protocol.json`.
Only detector preprocessing is covered here; the 30-minute tracking method and identity
decision are not yet selected.

The target is the last 30 minutes of the source video, native frames 49800:348600 (166 fps,
298800 frames). Because it contains known short failure intervals and the minute currently
running, it is not called unused identity-evaluation data. A geometry audit of the first frame
alone found direct support for 9 nodes of the 3 fish, but detector preprocessing reads none of
that pose, SLEAP, individual names or human annotations.

Applied to the long window, the existing fixed 9-frame background rule leaves only 2 frames.
So, as a separate rule, the 9 frames 5000,10000,...,45000, all before the whole target interval,
are fixed. The per-pixel median of the same 9 frames is passed to every chunk and every method.
No selection by frame appearance and no fallback on shortage are used. The background image is
not passed to the detector; it is saved as input for later background subtraction and the like.

The SHAs of the model, configuration, source video and original input manifest are registered,
and the existing batch1, threshold 0.5, FP32 parameters and backend settings are kept. All
detections of each image are kept as they are. The source MP4 is read through a read-only
hardlink; the video is not re-encoded or copied.

The 3 views are saved in 10-second units (1660 native frames). Only chunks with a completion
record for all 3 views and every output SHA are reused. Partial outputs and chunks with a
different protocol/code/model/input are not reused. An incomplete attempt is kept and a new
attempt is created; old files are not appended to. This is resumption of per-image independent
detection; no tracking initialization is run.

The final input is published only after the order of all chunks, contiguous native numbering,
all mask arrays and all raw-frame SHAs are consistent. Zero detections within a chunk are not
deleted either, and mask order and floating-point values are kept. Joining recomputes only the
offsets, and exact equality with arrays saved in a single pass over the same input is checked.

Before the full 30-minute GPU run, CPU tests, a comparison on a small real window, and reuse
after interruption are verified. The existence of this document and the JSON does not mean that
the 30-minute input is complete, that tracking is complete, or that there are no swaps.
