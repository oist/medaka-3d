"""B02 - distance-matched baselines and a quantified separability check.

Follows b01.  b01 showed that events differ from random baseline, but events only happen
when fish are close, so much of that is selection.  Here every usable event gets ONE baseline
window from the SAME experiment whose pre-context matches the event's on two distances:

    match variables  = mean of d(female, near male) and of d(male, male) over the 0.75 s that
                       precede the core (log scale), caliper |log ratio| <= 0.30 on both.
    core free        = the core overlaps no BORIS interval of ANY behaviour.
    circle baselines reuse the positive's own duration, so durations are matched exactly.

Positives with no baseline inside the caliper are DROPPED from the matched comparisons and
counted (that count is itself a finding: those events happen at distances the baseline never
reaches).  Matching removes "how close were they" as an explanation; what is left is dynamics.

Then each window is summarised (5 numbers x 12 features), and separability is measured with
cross-validation that never lets one fish group appear on both sides:

    K->P, P->K       (the only truly independent test available: 2 groups)
    leave-one-experiment-out (6 folds; experiments of one group are the SAME fish, so this
                       is optimistic and reported only as a secondary number)

Reference models: distance-only features (must sit near chance if matching worked), and a
duration-only model for attack-vs-circle.  Permutation p-values shuffle labels within experiment.

Nothing here is a claim about mate choice.  n_groups = 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b01_behavior_windows as B  # noqa: E402

warnings.filterwarnings("ignore")
FPS, PAD, NF = B.FPS, B.PAD, B.NF
PRE = int(round(0.75 * FPS))  # frames of context used for matching and for the "pre" summary
POST = int(round(0.5 * FPS))
A_PRE, A_POST = int(round(0.25 * FPS)), int(round(0.5 * FPS))  # attack core = [-0.25, +0.5] s
CALIPER = 0.30
I_DFN, I_DMM = 4, 6


# --------------------------------------------------------------------------------------
# windows with matched baselines
# --------------------------------------------------------------------------------------
def core_span(kind, c, dur):
    if kind == "attack":
        return c - A_PRE, c + A_POST
    return c, c + dur


def window_span(kind, c, dur):
    if kind == "attack":
        return c - PAD, c + PAD
    return c - PAD, c + dur + PAD


def make_window(kind, feat, c, dur):
    if kind == "attack":
        return B.onset_window(feat, c)
    return B.interval_window(feat, c, c + dur, B.N_EVENT)


def build(seed):
    rng = np.random.default_rng(seed)
    out = {"attack": [], "attack_neg": [], "circle": [], "circle_neg": [], "wrap": []}
    meta = []
    dropped = []
    for key in B.EXPERIMENTS:
        feat, segid, seg, pkg = B.load_experiment(key)
        ivs, pts = B.load_boris(key)
        T = len(feat)
        busy = B.busy_mask(T, ivs, pts)
        ivs["cls"] = ivs.beh.map(B.CLASS_OF)
        fin = np.isfinite(feat).all(1)
        cs_fin = np.concatenate([[0], np.cumsum(fin)])
        cs_busy = np.concatenate([[0], np.cumsum(busy)])
        ldfn = np.log(np.maximum(np.nan_to_num(feat[:, I_DFN], nan=1.0), 0.5))
        ldmm = np.log(np.maximum(np.nan_to_num(feat[:, I_DMM], nan=1.0), 0.5))
        cs_a = np.concatenate([[0], np.cumsum(ldfn)])
        cs_b = np.concatenate([[0], np.cumsum(ldmm)])
        used = np.zeros(T, bool)
        usable_idx = np.flatnonzero(segid >= 0)

        def pre_means(c0):
            """mean log distances over the PRE frames before core start c0 (array ok)."""
            a, b = c0 - PRE, c0
            return (cs_a[b] - cs_a[a]) / PRE, (cs_b[b] - cs_b[a]) / PRE

        def valid(a, b):
            ok = (a >= 0) & (b < T)
            a2, b2 = np.where(ok, a, 0), np.where(ok, b, 0)
            ok &= (segid[a2] >= 0) & (segid[a2] == segid[b2])
            ok &= (cs_fin[b2 + 1] - cs_fin[a2]) == (b2 - a2 + 1)
            return ok

        pos = ivs[ivs.cls.notna()].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        for r in pos.itertuples():
            kind = r.cls
            dur = 0 if kind == "attack" else int(r.f1 - r.f0)
            c = int(r.f0)
            ca, cb = core_span(kind, c, dur)
            wa, wb = window_span(kind, c, dur)
            ok = bool(valid(np.array([wa]), np.array([wb]))[0])
            m = dict(exp=key, kind=kind, beh=r.beh, subject=r.subject, f0=r.f0, f1=r.f1,
                     dur_s=(r.f1 - r.f0) / FPS, usable=ok,
                     other_events_in_window=B.other_events_in(ivs, wa, wb, True), matched=False)
            if not ok:
                meta.append(m)
                continue
            w = make_window(kind, feat, c, dur)
            if kind == "wrap":
                w = B.interval_window(feat, c, int(r.f1), B.N_WRAP)
                out["wrap"].append(w)
                m["row"] = len(out["wrap"]) - 1
                meta.append(m)
                continue
            pa, pb = pre_means(ca)
            # candidates: same experiment, own duration, core free, window valid, not already used
            cand = usable_idx[(usable_idx > PAD) & (usable_idx < T - dur - PAD - 2)]
            cca, ccb = core_span(kind, cand, dur)
            cwa, cwb = window_span(kind, cand, dur)
            good = valid(cwa, cwb) & ((cs_busy[ccb + 1] - cs_busy[cca]) == 0)
            good &= ~used[cand] & ~used[np.minimum(cand + dur, T - 1)]
            good &= (cca - PRE >= 0)
            cand = cand[good]
            if len(cand) == 0:
                meta.append(m)
                dropped.append((key, kind, "no candidate"))
                continue
            cca, _ = core_span(kind, cand, dur)
            qa, qb = pre_means(cca)
            err = np.maximum(np.abs(qa - pa), np.abs(qb - pb))
            j = int(np.argmin(err))
            if err[j] > CALIPER:
                meta.append(m)
                dropped.append((key, kind, f"caliper {err[j]:.2f}"))
                continue
            cn = int(cand[j])
            used[max(cn - PAD, 0): cn + dur + PAD + 1] = True
            out[kind].append(w)
            out[kind + "_neg"].append(make_window(kind, feat, cn, dur))
            m.update(row=len(out[kind]) - 1, matched=True, match_err=float(err[j]), neg_f0=cn)
            meta.append(m)
        print(f"{key}: matched attack {sum(1 for x in meta if x['exp']==key and x['kind']=='attack' and x['matched'])}"
              f" circle {sum(1 for x in meta if x['exp']==key and x['kind']=='circle' and x['matched'])}", flush=True)
    meta = pd.DataFrame(meta)
    arrays = {k: np.stack(v) if v else np.zeros((0, 0, NF), np.float32) for k, v in out.items()}
    return arrays, meta, dropped


# --------------------------------------------------------------------------------------
# summaries and classification
# --------------------------------------------------------------------------------------
STAT = ["pre_mean", "core_mean", "core_max", "core_min", "post_mean"]


def summarize(win, kind):
    """(n, L, NF) -> (n, 5*NF) with an identical definition for attack and circle windows."""
    if kind == "attack":
        pre = win[:, PAD - A_PRE - PRE: PAD - A_PRE]
        core = win[:, PAD - A_PRE: PAD + A_POST + 1]
        post = win[:, PAD + A_POST + 1: PAD + A_POST + 1 + POST]
    else:
        n = B.N_EVENT
        pre = win[:, PAD - PRE: PAD]
        core = win[:, PAD: PAD + n]
        post = win[:, PAD + n: PAD + n + POST]
    s = np.stack([pre.mean(1), core.mean(1), core.max(1), core.min(1), post.mean(1)], axis=1)  # (n,5,NF)
    return s.reshape(len(win), -1)


def colnames():
    return [f"{B.FEATURES[i][0]}:{s}" for s in STAT for i in range(NF)]  # matches reshape order below


def reorder(s):
    """summarize() flattens (stat, feat) -> index stat*NF+feat; keep that, names follow."""
    return s


def cols_for(stats, feats):
    return [si * NF + fi for si in stats for fi in feats]


# index into STAT: 0 pre_mean 1 core_mean 2 core_max 3 core_min 4 post_mean
FEATURE_SETS = {
    # the two matched variables measured BEFORE the core: must be near chance if matching worked
    "matched-vars-pre-only": cols_for([0], (I_DFN, I_DMM)),
    # distance geometry incl. what happens during/after the core (closing, separating)
    "distance-dynamics-only": cols_for(range(5), (4, 5, 6, 7, 8)),
    # no distance or closing features at all: speeds and yaw rates only
    "kinematics-only": cols_for(range(5), (0, 1, 2, 3, 9)),
}


def models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")),
        "RF": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=0),
    }


def fold_scores(X, y, group, exp, scheme, model, binary, positive=None):
    yp = np.empty(len(y), dtype=object)
    prob = np.full(len(y), np.nan)
    if scheme == "K->P":
        folds = [(group == "K", group == "P")]
    elif scheme == "P->K":
        folds = [(group == "P", group == "K")]
    else:
        folds = [(exp != e, exp == e) for e in np.unique(exp)]
    mask = np.zeros(len(y), bool)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        m = _clone(model)
        m.fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
        mask |= te
        if binary:
            col = list(m.classes_).index(positive)
            prob[te] = m.predict_proba(X[te])[:, col]
    return yp, prob, mask


def _clone(m):
    from sklearn.base import clone
    return clone(m)


def evaluate(X, y, group, exp, scheme, model, binary, positive=None):
    yp, prob, mask = fold_scores(X, y, group, exp, scheme, model, binary, positive)
    if not mask.any():
        return dict(bal_acc=np.nan, auc=np.nan, n=0)
    ba = balanced_accuracy_score(np.asarray(y[mask]).astype(str), yp[mask].astype(str))
    auc = np.nan
    if binary:
        yy = (y[mask] == positive).astype(int)
        if 0 < yy.sum() < len(yy):
            auc = roc_auc_score(yy, prob[mask])
    return dict(bal_acc=ba, auc=auc, n=int(mask.sum()))


def perm_p(X, y, group, exp, model, binary, positive, observed, n_perm, rng):
    """permute labels WITHIN experiment, leave-one-experiment-out scheme."""
    hits = 0
    for _ in range(n_perm):
        yp = y.copy()
        for e in np.unique(exp):
            idx = np.flatnonzero(exp == e)
            yp[idx] = rng.permutation(y[idx])
        r = evaluate(X, yp, group, exp, "LOEO", model, binary, positive)
        hits += r["bal_acc"] >= observed
    return (hits + 1) / (n_perm + 1)


def run_task(name, X, y, group, exp, cols, binary, positive, n_perm, rng):
    rows = []
    for mname, model in models().items():
        for fname, feats in (("all", cols), *FEATURE_SETS.items()):
            if mname == "RF" and fname == "matched-vars-pre-only":
                continue
            for scheme in ("K->P", "P->K", "LOEO"):
                r = evaluate(X[:, feats], y, group, exp, scheme, model, binary, positive)
                p = np.nan
                if scheme == "LOEO" and mname == "LR" and n_perm > 0 and fname in ("all", "kinematics-only"):
                    p = perm_p(X[:, feats], y, group, exp, model, binary, positive, r["bal_acc"], n_perm, rng)
                rows.append(dict(task=name, model=mname, features=fname, scheme=scheme, **r, perm_p=p))
                print(f"  {name:18s} {mname:2s} {fname:13s} {scheme:5s} bal_acc={r['bal_acc']:.3f} auc={r['auc']:.3f} n={r['n']}",
                      flush=True)
    return rows


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("work/b02_matched_separability"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=100)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed + 1)

    arrays, meta, dropped = build(a.seed)
    np.savez_compressed(a.out / "windows_matched.npz", **arrays)
    meta.to_csv(a.out / "events_matched.csv", index=False)

    ev = meta[meta.kind.isin(["attack", "circle"])]
    tab = ev.groupby("kind").agg(events=("usable", "size"), usable=("usable", "sum"), matched=("matched", "sum"))
    print(tab.to_string())
    print("match error (max abs log ratio) quantiles:",
          np.percentile(ev[ev.matched].match_err, [50, 90, 100]).round(3).tolist())

    # ---- curves / heatmaps with matched baselines
    mm = meta.copy()
    for kind in ("attack", "circle"):
        sel = mm[(mm.kind == kind) & mm.matched].sort_values(["exp", "dur_s"])
        order = sel.row.values.astype(int)
        arrays_sorted = {kind: arrays[kind][order], kind + "_neg": arrays[kind + "_neg"][order]}
        m2 = sel.copy()
        m2["row"] = np.arange(len(sel))
        m2["usable"] = True
        B.curve_figure(arrays_sorted, m2, kind, kind + "_neg", a.out / f"curves_matched_{kind}.png", neg_label="matched baseline")
        B.heatmap_figure(arrays_sorted, m2, kind, kind + "_neg", a.out / f"heatmap_matched_{kind}.png", neg_label="matched baseline")

    # ---- summaries
    blocks, labels = [], []
    for kind in ("attack", "circle"):
        sel = mm[(mm.kind == kind) & mm.matched]
        order = sel.row.values.astype(int)
        for lab, arr in ((kind, arrays[kind]), ("baseline_" + kind, arrays[kind + "_neg"])):
            X = summarize(arr[order], kind)
            blocks.append(pd.DataFrame(X).assign(label=lab, exp=sel.exp.values, dur=sel.dur_s.values,
                                                 kind_of_window=kind))
    D = pd.concat(blocks, ignore_index=True)
    D["group"] = D.exp.str[0]
    feat_cols = list(range(5 * NF))
    X = D[feat_cols].values.astype(float)
    names = [f"{B.FEATURES[fi][0]}:{STAT[si]}" for si in range(5) for fi in range(NF)]

    results = []
    # 1 attack vs its matched baseline
    s = D.kind_of_window == "attack"
    results += run_task("attack_vs_baseline", X[s], D.label[s].values, D.group[s].values, D.exp[s].values,
                        feat_cols, True, "attack", a.n_perm, rng)
    s = D.kind_of_window == "circle"
    results += run_task("circle_vs_baseline", X[s], D.label[s].values, D.group[s].values, D.exp[s].values,
                        feat_cols, True, "circle", a.n_perm, rng)
    # 3 attack vs circle (positives only) - includes a duration-only reference below
    s = D.label.isin(["attack", "circle"])
    results += run_task("attack_vs_circle", X[s], D.label[s].values, D.group[s].values, D.exp[s].values,
                        feat_cols, True, "attack", a.n_perm, rng)
    Xd = D.loc[s, ["dur"]].values.astype(float)
    for scheme in ("K->P", "P->K", "LOEO"):
        r = evaluate(Xd, D.label[s].values, D.group[s].values, D.exp[s].values, scheme, models()["LR"], True, "attack")
        results.append(dict(task="attack_vs_circle", model="LR", features="duration-only", scheme=scheme, **r, perm_p=np.nan))
        print(f"  attack_vs_circle duration-only {scheme} bal_acc={r['bal_acc']:.3f}")
    # 4 three-class
    y3 = D.label.map(lambda v: "baseline" if v.startswith("baseline") else v).values
    for mname, model in models().items():
        for scheme in ("K->P", "P->K", "LOEO"):
            yp, _, mask = fold_scores(X, y3, D.group.values, D.exp.values, scheme, model, False)
            ba = balanced_accuracy_score(y3[mask], yp[mask].astype(str))
            cm = confusion_matrix(y3[mask], yp[mask].astype(str), labels=["attack", "circle", "baseline"])
            results.append(dict(task="three_class", model=mname, features="all", scheme=scheme, bal_acc=ba,
                                auc=np.nan, n=int(mask.sum()), perm_p=np.nan, confusion=cm.tolist()))
            print(f"  three_class {mname} {scheme} bal_acc={ba:.3f} confusion(rows=true attack,circle,baseline)={cm.tolist()}")
    pd.DataFrame(results).to_csv(a.out / "separability.csv", index=False)

    # ---- top features (LR coefficients standardized, attack vs matched baseline & circle vs baseline)
    top = {}
    for kind in ("attack", "circle"):
        s = (D.kind_of_window == kind).values
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        m.fit(X[s], (D.label[s] == kind).astype(int))
        coef = m[-1].coef_[0]
        o = np.argsort(-np.abs(coef))[:8]
        top[kind] = [(names[i], round(float(coef[i]), 2)) for i in o]
        print(kind, "top |coef|:", top[kind])
    (a.out / "top_features.json").write_text(json.dumps(top, indent=2))

    # ---- PCA 3D and LDA, 3-class set
    y3s = pd.Series(y3)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(3).fit(Z)
    P = pca.transform(Z)
    lda = LinearDiscriminantAnalysis(n_components=2).fit(Z, y3)
    L = lda.transform(Z)
    colour = {"attack": "tab:red", "circle": "tab:blue", "baseline": "0.55"}
    marker = {"K": "o", "P": "^"}
    fig = plt.figure(figsize=(18, 6))
    for k, (elev, azim, title) in enumerate([(20, 40, "PCA 3D, view 1"), (20, 130, "PCA 3D, view 2")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for c, g in product(colour, marker):
            sel = ((y3s == c) & (D.group == g)).values
            ax.scatter(P[sel, 0], P[sel, 1], P[sel, 2], s=14, c=colour[c], marker=marker[g], alpha=0.65,
                       label=f"{c} / {g}")
        ax.view_init(elev, azim)
        ax.set_xlabel(f"PC1 {pca.explained_variance_ratio_[0]:.0%}")
        ax.set_ylabel(f"PC2 {pca.explained_variance_ratio_[1]:.0%}")
        ax.set_zlabel(f"PC3 {pca.explained_variance_ratio_[2]:.0%}")
        ax.set_title(title)
        if k == 0:
            ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 3, 3)
    for c, g in product(colour, marker):
        sel = ((y3s == c) & (D.group == g)).values
        ax.scatter(L[sel, 0], L[sel, 1], s=14, c=colour[c], marker=marker[g], alpha=0.65)
    ax.set_title("LDA 2D (supervised, fitted on all data: optimistic, for display only)")
    fig.suptitle("matched-baseline windows; colour = class, marker = fish group (o K, ^ P)")
    fig.tight_layout()
    fig.savefig(a.out / "pca3d_lda.png", dpi=100)
    plt.close(fig)

    prov = dict(tool="tools/b02_matched_separability.py", seed=a.seed, n_perm=a.n_perm, caliper=CALIPER,
                match_vars=["log d_F_Mnear", "log d_M_M"], pre_s=0.75, post_s=0.5,
                dropped=[list(x) for x in dropped], counts=tab.reset_index().to_dict("records"),
                summary_columns=names, b01_provenance="see work/b01_behavior_windows/provenance.json")
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print("wrote", a.out.resolve())


if __name__ == "__main__":
    main()
