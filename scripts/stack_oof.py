#!/usr/bin/env python3
"""Per-label stacking weights over multiple models' out-of-fold predictions.

    python3 scripts/stack_oof.py models/kaggle_traing10_v5/models/slotknee_g10 \
        models/slotknee_t4/models/slotknee_full5 --out models/stack_weights.json

Rank-averaging with EQUAL weights leaves AUC on the table when models have
per-label strengths (e.g. a notch-zoom view for ACL, a wide view for OA).  This
fits, for each label, convex weights over the models that maximise the AUC of the
weighted rank-average — validated honestly: weights are fitted on four outer
folds' rows and scored on the held-out fold's rows (the OOF row's own fold), so
the reported gain is out-of-sample.  Final weights are refitted on all rows.

Models may sit on different fold splits (rows are aligned by StudyInstanceUID;
the FIRST directory's fold assignment drives the internal split).  Labels where
the fitted weights do not beat equal weights out-of-sample fall back to equal.

Output json: {"labels": {label: [w_per_model]}, "models": [dirs], "report": {...}}
The submit kernel can consume it later; until then this is a measurement tool.
"""
import argparse
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.llm_labels import LABELS


def load_oof(d):
    """{uid: (prob_vector, y, w, fold)} from a run dir (recursive)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "oof_fold_*.npz"), recursive=True)):
        k = int(os.path.basename(p).split("_")[2].split(".")[0])
        z = np.load(p, allow_pickle=True)
        for uid, lg, y, w in zip(z["uids"], z["logits"], z["y"], z["w"]):
            out[str(uid)] = (1.0 / (1.0 + np.exp(-np.asarray(lg, float))),
                             np.asarray(y, float), np.asarray(w, float), k)
    return out


def weight_grid(n, step=0.1):
    """Convex weight vectors over n models on a coarse simplex grid."""
    ticks = np.arange(0.0, 1.0 + 1e-9, step)
    for combo in itertools.product(ticks, repeat=n - 1):
        if sum(combo) <= 1.0 + 1e-9:
            yield np.array(list(combo) + [1.0 - sum(combo)])


def auc_of(weights, ranks, yb, m):
    s = sum(w * r for w, r in zip(weights, ranks))
    return roc_auc_score(yb[m], s[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="two or more OOF run dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--prefixes", nargs="+", default=None,
                    help="explicit per-dir prefixes for the submit kernel (overrides filename-derived ones; needed when source dirs hold bare fold_k files)")
    a = ap.parse_args()
    runs = [load_oof(d) for d in a.dirs]
    uids = sorted(set.intersection(*[set(r) for r in runs]))
    print(f"models: {len(runs)}  common studies: {len(uids)}")
    y = np.stack([runs[0][u][1] for u in uids])
    w = np.stack([runs[0][u][2] for u in uids])
    folds = np.array([runs[0][u][3] for u in uids])
    probs = [np.stack([r[u][0] for u in uids]) for r in runs]

    weights, report = {}, {}
    equal = np.ones(len(runs)) / len(runs)
    for li, lab in enumerate(LABELS):
        m_all = w[:, li] > 0
        yb = (y[:, li] > 0.5).astype(int)
        ranks = [rankdata(p[:, li]) / len(uids) for p in probs]
        # honest evaluation: fit on 4 folds, score on the held-out fold
        oos_fit, oos_eq = [], []
        for k in sorted(set(folds)):
            tr = m_all & (folds != k)
            te = m_all & (folds == k)
            if len(np.unique(yb[tr])) < 2 or len(np.unique(yb[te])) < 2:
                continue
            best = max(weight_grid(len(runs), a.step), key=lambda ww: auc_of(ww, ranks, yb, tr))
            oos_fit.append(auc_of(best, ranks, yb, te))
            oos_eq.append(auc_of(equal, ranks, yb, te))
        gain = float(np.mean(oos_fit) - np.mean(oos_eq)) if oos_fit else 0.0
        if gain > 0:
            final = max(weight_grid(len(runs), a.step), key=lambda ww: auc_of(ww, ranks, yb, m_all))
        else:
            final = equal
        weights[lab] = [round(float(x), 3) for x in final]
        report[lab] = {"oos_gain_vs_equal": round(gain, 4), "kept": bool(gain > 0)}
        print(f"  {lab:18s} weights={weights[lab]}  oos gain vs equal {gain:+.4f} {'KEPT' if gain > 0 else 'equal'}")
    macro_gain = float(np.mean([r["oos_gain_vs_equal"] for r in report.values() if r["kept"]] or [0.0]))
    # The submit kernel matches weights to layout groups by CHECKPOINT FILENAME PREFIX
    # ('g10t1_fold_0_best.pt' -> 'g10t1'; bare 'fold_0_best.pt' -> '').  Derive each dir's
    # prefix from its own files so the tool and the kernel agree by construction.  Prefixes
    # must stay layout-level (both seeds of one layout share a prefix).
    def dir_prefix(d):
        for p in sorted(glob.glob(os.path.join(d, "**", "*_best.pt"), recursive=True)):
            b = os.path.basename(p); i = b.find("fold_")
            return b[:i].rstrip("_") if i > 0 else ""
        return ""
    prefixes = list(a.prefixes) if a.prefixes else [dir_prefix(d) for d in a.dirs]
    assert len(prefixes) == len(a.dirs), "--prefixes must match the number of dirs"
    if len(set(prefixes)) != len(prefixes):
        print(f"WARNING: non-unique prefixes {prefixes} — the submit kernel cannot tell these groups apart")
    out = {"models": a.dirs, "prefixes": prefixes, "labels": weights, "report": report,
           "note": "weights fitted per label on OOF ranks; labels without out-of-sample gain use equal weights; prefixes key the submit kernel's layout groups"}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {a.out}; mean out-of-sample gain on kept labels: {macro_gain:+.4f}")


if __name__ == "__main__":
    main()
