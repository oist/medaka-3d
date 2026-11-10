"""B01 - behaviour windows: curves and heatmaps of 3D kinematic features around BORIS events.

Purpose (exploratory, descriptive; nothing here is a result about mate choice)
    Do the hand-annotated aggressive / courtship-circling / wrapping events differ from
    quiet baseline in simple 3D kinematic features?  This tool only builds the windows and
    draws them.  It does NOT classify.

Design decisions, all agreed with the user before writing:
    * Three classes.  attack = K "Dash" + P "attack" + P "attempted att";
      circle = K "Courtship" + P "quick circle" + P "incomplete qc"; wrap = K/P "Wrap".
    * attack events last 0.04-0.4 s, i.e. they are point-like: ONSET-LOCKED window of
      -1 s .. +1 s, no resampling.
    * circle / wrap events are intervals: 1 s real-time pre-context, the event resampled
      to N points, 1 s real-time post-context.  Duration is kept as a separate column.
    * Only frames inside the reviewer's usable segments (`id_pass/id_segments.csv`) count.
      A window must lie wholly inside ONE segment, because the F/M1/M2 naming changes
      between segments.  Frames before `analysis_start` are not in any segment.
    * Negatives: windows whose CORE (attack: onset +-0.25 s; circle: the event part) overlaps
      no BORIS interval of ANY behaviour (follow, approach, chase ... included).  Context is
      not constrained; `context_free` records whether it was.
    * wrap has only a handful of usable events: curves are drawn per event, no statistics.

Features are role-free on purpose (BORIS m1/m2 are NOT mapped to appearance M1/M2, see
docs/PRIOR_FINDINGS.md section 1.5).  The two males are ordered per frame as "near" / "far"
by distance to the female.

Run:  <python with numpy scipy pandas matplotlib> tools/b01_behavior_windows.py
Output: work/b01_behavior_windows/ (gitignored) with windows.npz, events.csv, figures, provenance.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

FPS = 166.0
PAD = int(round(1.0 * FPS))  # 1 s context
N_EVENT = 30  # samples for the resampled event part (circle)
N_WRAP = 60  # samples for wrap (events are much longer)
CORE_ATTACK = int(round(0.25 * FPS))
SG_WIN = 17  # ~0.1 s
SG_POLY = 3

REVIEW_ROOT = Path("Z:/medaka/review_packages")
RECON_ROOT = Path("Z:/medaka/reconstruction_3d")
BORIS_K = Path("C:/Users/ReiterU/Documents/GitHub/medaka-mating-pipeline/archive/boris")
BORIS_P = BORIS_K / "p2"

EXPERIMENTS = {
    # key: (experiment folder, review package tag, BORIS csv)
    "K_1_1": ("20251225_K_1_1_FCMC", "s16_bottomup_3node_uncapped_m3dfull", BORIS_K / "20251225_K_1_1_FCMC.csv"),
    "K_1_2": ("20251225_K_1_2_FCMC", "s16_bottomup_3node_uncapped_m3dfull", BORIS_K / "20251225_K_1_2_FCMC.csv"),
    "K_1_3": ("20251225_K_1_3_FCMC", "s16_bottomup_3node_uncapped_m3dfull", BORIS_K / "20251225_K_1_3_FCMC.csv"),
    "P_1_1": ("20260112_P_1_1_FCMC", "s14_bottomup_3node_reviewed_m3dfull", BORIS_P / "20260112_P_1_1_FCMC.csv"),
    "P_1_2": ("20260113_P_1_2_FCMC", "s14_bottomup_3node_reviewed_m3dfull", BORIS_P / "20260112_P_1_2_FCMC.csv"),
    "P_1_3": ("20260112_P_1_3_FCMC", "s14_bottomup_3node_reviewed_m3dfull", BORIS_P / "20260112_P_1_3_FCMC.csv"),
}

CLASS_OF = {
    "dash": "attack", "attack": "attack", "attempted att": "attack",
    "courtship": "circle", "quick circle": "circle", "incomplete qc": "circle",
    "wrap": "wrap",
}

FEATURES = [
    ("F_speed", "female speed (mm/s)"),
    ("F_yawrate", "female |yaw rate| (deg/s)"),
    ("Mnear_speed", "near-male speed (mm/s)"),
    ("Mfar_speed", "far-male speed (mm/s)"),
    ("d_F_Mnear", "female - near male (mm)"),
    ("d_F_Mfar", "female - far male (mm)"),
    ("d_M_M", "male - male (mm)"),
    ("close_F_Mnear", "closing speed F-near (mm/s, <0 = approaching)"),
    ("close_M_M", "closing speed M-M (mm/s, <0 = approaching)"),
    ("Mnear_yawrate", "near-male |yaw rate| (deg/s)"),
    ("Mnear_head_vs_F", "near-male heading vs line to F (deg)"),
    ("F_head_vs_Mnear", "female heading vs line to near male (deg)"),
]
NF = len(FEATURES)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def parse_names(s: str) -> dict[str, int]:
    out = {}
    for tok in s.split():
        role, fid = tok.split("=")
        out[role] = int(fid.split("_")[1]) - 1
    return out


def sg(x, deriv=0):
    return savgol_filter(x, SG_WIN, SG_POLY, deriv=deriv, delta=1.0 / FPS, axis=0, mode="interp")


def angle_between(a, b):
    """abs angle (deg) between 2D vectors a, b (..., 2)."""
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = (a * b).sum(-1)
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def segment_features(nodes, head, tail, roles):
    """nodes (n, 3 fish, 3 nodes, 3) of one segment -> (n, NF)."""
    cen = nodes.mean(axis=2)  # (n, fish, 3)
    f, m1, m2 = roles["F"], roles["M1"], roles["M2"]
    cF, cA, cB = cen[:, f], cen[:, m1], cen[:, m2]
    dA, dB = np.linalg.norm(cF - cA, axis=1), np.linalg.norm(cF - cB, axis=1)
    nearA = dA <= dB
    cN, cX = np.where(nearA[:, None], cA, cB), np.where(nearA[:, None], cB, cA)
    fishN = np.where(nearA, m1, m2)
    vF = np.linalg.norm(sg(cF, 1), axis=1)
    vN = np.linalg.norm(sg(cN, 1), axis=1)
    vX = np.linalg.norm(sg(cX, 1), axis=1)
    dFN = np.linalg.norm(cF - cN, axis=1)
    dFX = np.linalg.norm(cF - cX, axis=1)
    dMM = np.linalg.norm(cA - cB, axis=1)

    def heading(fish_idx):
        idx = np.arange(len(nodes))
        fi = np.broadcast_to(fish_idx, idx.shape)
        h = nodes[idx, fi, head, :2] - nodes[idx, fi, tail, :2]
        return h

    hF = heading(f)
    hN = heading(fishN)

    def yawrate(h):
        yaw = np.unwrap(np.arctan2(h[:, 1], h[:, 0]))
        return np.degrees(np.abs(sg(yaw, 1)))

    out = np.stack([
        vF, yawrate(hF), vN, vX, dFN, dFX, dMM,
        sg(dFN, 1), sg(dMM, 1), yawrate(hN),
        angle_between(hN, cF[:, :2] - cN[:, :2]),
        angle_between(hF, cN[:, :2] - cF[:, :2]),
    ], axis=1)
    return out.astype(np.float32)


def load_experiment(key):
    exp, tag, boris_csv = EXPERIMENTS[key]
    pkg = REVIEW_ROOT / exp / tag
    seg = pd.read_csv(pkg / "id_pass" / "id_segments.csv")
    z = np.load(pkg / "review_data.npz", allow_pickle=True)
    nodes = z["nodes3d_filled"]
    head, tail = int(z["head_node"]), int(z["tail_node"])
    T = nodes.shape[0]
    feat = np.full((T, NF), np.nan, np.float32)
    segid = np.full(T, -1, np.int32)
    for i, r in seg.iterrows():
        a, b = int(r.start), int(r.end) + 1
        feat[a:b] = segment_features(nodes[a:b], head, tail, parse_names(r.names))
        segid[a:b] = i
    return feat, segid, seg, pkg


def load_boris(key):
    _, _, csv = EXPERIMENTS[key]
    d = pd.read_csv(csv)
    d["b"] = d["Behavior"].str.lower().str.strip()
    ivs = []  # every START/STOP interval, any behaviour: (b, subject, f0, f1)
    for (b, subj), g in d[d["Behavior type"].isin(["START", "STOP"])].groupby(["b", "Subject"]):
        g = g.sort_values("Time")
        s = g[g["Behavior type"] == "START"]["Time"].values
        t = g[g["Behavior type"] == "STOP"]["Time"].values
        n = min(len(s), len(t))
        for x, y in zip(s[:n], t[:n]):
            ivs.append((b, subj, int(round(x * FPS)), int(round(y * FPS))))
    pts = d[d["Behavior type"] == "POINT"]
    pts = [(r.b, int(round(r.Time * FPS))) for r in pts.itertuples()]
    return pd.DataFrame(ivs, columns=["beh", "subject", "f0", "f1"]), pts


# --------------------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------------------
def inside_one_segment(segid, a, b):
    """frames a..b inclusive all in the same usable segment."""
    if a < 0 or b >= len(segid):
        return False
    return segid[a] >= 0 and segid[a] == segid[b]


def onset_window(feat, f0):
    return feat[f0 - PAD: f0 + PAD + 1]


def interval_window(feat, f0, f1, n_event):
    pre = feat[f0 - PAD: f0]
    post = feat[f1 + 1: f1 + 1 + PAD]
    t = np.linspace(f0, f1, n_event)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, f1)
    w = (t - lo)[:, None]
    mid = feat[lo] * (1 - w) + feat[hi] * w
    return np.concatenate([pre, mid, post], axis=0)


def busy_mask(T, ivs, pts):
    m = np.zeros(T, bool)
    for r in ivs.itertuples():
        m[max(r.f0, 0): min(r.f1 + 1, T)] = True
    for _, f in pts:
        m[max(f - 5, 0): min(f + 6, T)] = True
    return m


def other_events_in(ivs, a, b, skip):
    o = ivs[(ivs.f1 >= a) & (ivs.f0 <= b)]
    return int(len(o) - (1 if skip else 0))


def build(args):
    rng = np.random.default_rng(args.seed)
    win = {"attack": [], "circle": [], "wrap": [], "attack_neg": [], "circle_neg": []}
    meta = []
    for key in EXPERIMENTS:
        feat, segid, seg, pkg = load_experiment(key)
        ivs, pts = load_boris(key)
        T = len(feat)
        busy = busy_mask(T, ivs, pts)
        ivs["cls"] = ivs.beh.map(CLASS_OF)
        circle_durs = []
        for r in ivs[ivs.cls.notna()].itertuples():
            nev = N_WRAP if r.cls == "wrap" else N_EVENT
            if r.cls == "attack":
                a, b = r.f0 - PAD, r.f0 + PAD
            else:
                a, b = r.f0 - PAD, r.f1 + PAD
            ok = inside_one_segment(segid, a, b) and np.isfinite(feat[a:b + 1]).all()
            m = dict(exp=key, kind=r.cls, beh=r.beh, subject=r.subject, f0=r.f0, f1=r.f1,
                     dur_s=(r.f1 - r.f0) / FPS, usable=ok, other_events_in_window=other_events_in(ivs, a, b, True))
            if ok:
                w = onset_window(feat, r.f0) if r.cls == "attack" else interval_window(
                    feat, r.f0, r.f1, N_WRAP if r.cls == "wrap" else N_EVENT)
                m["row"] = len(win[r.cls])
                win[r.cls].append(w)
                if r.cls == "circle":
                    circle_durs.append(r.f1 - r.f0)
            meta.append(m)
        # negatives: core free of any BORIS interval, whole window inside one usable segment
        usable_frames = np.flatnonzero(segid >= 0)
        for kind, per_exp in (("attack_neg", args.n_neg), ("circle_neg", args.n_neg)):
            got, tries = 0, 0
            while got < per_exp and tries < 200000:
                tries += 1
                c = int(rng.choice(usable_frames))
                if kind == "attack_neg":
                    a, b = c - PAD, c + PAD
                    core = slice(c - CORE_ATTACK, c + CORE_ATTACK + 1)
                    dur = 0
                else:
                    dur = int(rng.choice(circle_durs)) if circle_durs else int(1.5 * FPS)
                    a, b = c - PAD, c + dur + PAD
                    core = slice(c, c + dur + 1)
                if not inside_one_segment(segid, a, b) or not np.isfinite(feat[a:b + 1]).all():
                    continue
                if busy[core].any():
                    continue
                w = onset_window(feat, c) if kind == "attack_neg" else interval_window(feat, c, c + dur, N_EVENT)
                ctx_free = not busy[a:b + 1].any()
                meta.append(dict(exp=key, kind=kind, beh="", subject="", f0=c, f1=c + dur, dur_s=dur / FPS,
                                 usable=True, other_events_in_window=int(not ctx_free), context_free=ctx_free,
                                 row=len(win[kind])))
                win[kind].append(w)
                got += 1
            print(f"{key} {kind}: {got} negatives ({tries} tries)", flush=True)
    meta = pd.DataFrame(meta)
    arrays = {k: np.stack(v) if v else np.zeros((0, 0, NF), np.float32) for k, v in win.items()}
    return arrays, meta


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
COL = {"attack": "tab:red", "circle": "tab:blue", "wrap": "tab:green", "neg": "0.45"}


def x_axes(kind):
    if kind == "attack":
        return (np.arange(2 * PAD + 1) - PAD) / FPS, None
    n = N_WRAP if kind == "wrap" else N_EVENT
    x = np.arange(PAD + n + PAD)
    return x, (PAD, PAD + n)


def curve_figure(arrays, meta, kind, neg_kind, path, per_event=False, neg_label="baseline"):
    x, bounds = x_axes(kind)
    pos = arrays[kind]
    neg = arrays[neg_kind] if neg_kind else None
    fig, axs = plt.subplots(4, 3, figsize=(16, 13), sharex=True)
    for i, ax in enumerate(axs.ravel()):
        if per_event:
            for w in pos:
                ax.plot(x, w[:, i], lw=0.8, alpha=0.7, color=COL[kind])
        else:
            for arr, col, lab in ((neg, COL["neg"], f"{neg_label} (n={0 if neg is None else len(neg)})"),
                                  (pos, COL[kind], f"{kind} (n={len(pos)})")):
                if arr is None or len(arr) == 0:
                    continue
                med = np.nanmedian(arr[:, :, i], axis=0)
                q1, q3 = np.nanpercentile(arr[:, :, i], [25, 75], axis=0)
                ax.fill_between(x, q1, q3, color=col, alpha=0.25, lw=0)
                ax.plot(x, med, color=col, lw=1.6, label=lab)
        if bounds:
            for b in bounds:
                ax.axvline(b, color="k", lw=0.6, ls="--")
        else:
            ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(FEATURES[i][1], fontsize=9)
    if bounds:
        axs[-1, 1].set_xlabel("1 s pre (real time) | event resampled | 1 s post (real time)")
    else:
        axs[-1, 1].set_xlabel("time from onset (s)")
    if not per_event:
        axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{kind}: median and IQR (medaka-3d b01; exploratory)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def heatmap_figure(arrays, meta, kind, neg_kind, path, neg_label="baseline"):
    x, bounds = x_axes(kind)
    m = meta[(meta.kind == kind) & meta.usable].copy().sort_values(["exp", "dur_s"])
    pos = arrays[kind][m.row.values.astype(int)]
    neg = arrays[neg_kind][: len(pos) or 1] if neg_kind else None
    cols = 2 if neg is not None else 1
    fig, axs = plt.subplots(NF, cols, figsize=(6 * cols + 1.5, 3.0 * NF), squeeze=False, constrained_layout=True)
    for i in range(NF):
        allv = pos[:, :, i] if neg is None else np.concatenate([pos[:, :, i], neg[:, :, i]])
        lo, hi = np.nanpercentile(allv, [2, 98])
        for j, (arr, name) in enumerate([(pos, kind), (neg, neg_label)][:cols]):
            ax = axs[i, j]
            im = ax.imshow(arr[:, :, i], aspect="auto", vmin=lo, vmax=hi, cmap="viridis",
                           extent=[x[0], x[-1], len(arr), 0], interpolation="nearest")
            if bounds:
                for b in bounds:
                    ax.axvline(b, color="w", lw=0.6, ls="--")
            else:
                ax.axvline(0, color="w", lw=0.6, ls="--")
            ax.set_title(f"{FEATURES[i][1]} - {name}", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axs[i].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle(f"{kind}: one row per event, sorted by experiment then duration; colour = feature (2-98 pct)")
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("work/b01_behavior_windows"))
    p.add_argument("--n-neg", type=int, default=80, help="negatives per experiment per window type")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arrays, meta = build(args)
    np.savez_compressed(args.out / "windows.npz", **arrays)
    meta.to_csv(args.out / "events.csv", index=False)

    ev = meta[~meta.kind.str.endswith("_neg")]
    summary = ev.groupby(["kind", "exp"]).agg(events=("usable", "size"), usable=("usable", "sum")).unstack(fill_value=0)
    print(summary.to_string())

    curve_figure(arrays, meta, "attack", "attack_neg", args.out / "curves_attack.png")
    curve_figure(arrays, meta, "circle", "circle_neg", args.out / "curves_circle.png")
    curve_figure(arrays, meta, "wrap", None, args.out / "curves_wrap_per_event.png", per_event=True)
    heatmap_figure(arrays, meta, "attack", "attack_neg", args.out / "heatmap_attack.png")
    heatmap_figure(arrays, meta, "circle", "circle_neg", args.out / "heatmap_circle.png")
    heatmap_figure(arrays, meta, "wrap", None, args.out / "heatmap_wrap.png")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    prov = dict(
        tool="tools/b01_behavior_windows.py", created=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv), python=platform.python_version(), git_head=sha,
        params=dict(fps=FPS, pad_s=1.0, n_event=N_EVENT, n_wrap=N_WRAP, core_attack_s=0.25,
                    savgol_window=SG_WIN, savgol_poly=SG_POLY, n_neg=args.n_neg, seed=args.seed),
        experiments={k: dict(folder=v[0], package=v[1], boris=str(v[2])) for k, v in EXPERIMENTS.items()},
        features=[f[0] for f in FEATURES],
        counts={f"{k}": int(len(v)) for k, v in arrays.items()},
    )
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print("wrote", args.out.resolve())


if __name__ == "__main__":
    main()
