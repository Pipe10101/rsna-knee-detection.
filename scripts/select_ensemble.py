#!/usr/bin/env python3
"""Greedy forward selection over OOF runs: which models should the ensemble contain?

    python3 scripts/select_ensemble.py models/ablate_full/*_s42 models/kaggle_traing10_v5/... \
        --out models/ensemble_choice.json [--per-label] [--max 6]

WHY THIS EXISTS.  With capacity, resolution, anchor count and seeds all measured out
(2026-08-24), the largest remaining lever is a DIVERSE ensemble stacked per label — the
standard path to the top of a leaderboard, and the one axis our measurements do NOT say is
exhausted.  But "throw every checkpoint in" is measurably wrong here: the 10-member two-view
entry scored 0.864 against 0.866 for a clean 2-member one, because weak members dilute a
rank-average.  Selection is the difference between those two outcomes.

METHOD.  Greedy forward selection WITH REPLACEMENT (Caruana et al. 2004): repeatedly add the
model that most improves the rank-averaged macro AUC, allowing the same model to be picked
again (which is how a greedy ensemble expresses weights).  Scored honestly: the selection is
run on four of the five OOF folds and evaluated on the held-out fold, rotated, so the reported
gain is out-of-sample.  A final selection is then refit on everything.

Only models whose OOF covers the SAME studies are comparable, so runs are intersected on
StudyInstanceUID and the count is printed — a small intersection means the comparison is weak,
not that the ensemble is good.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.llm_labels import LABELS


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "oof_fold_*.npz"), recursive=True)):
        k = int(os.path.basename(p).split("_")[2].split(".")[0])
        z = np.load(p, allow_pickle=True)
        for uid, lg, y, w in zip(z["uids"], z["logits"], z["y"], z["w"]):
            out[str(uid)] = (np.asarray(lg, float), np.asarray(y, float), np.asarray(w, float), k)
    return out


def macro(score, Y, W, cols=None):
    aucs = []
    for j in (cols if cols is not None else range(len(LABELS))):
        m = W[:, j] > 0
        yb = (Y[m, j] > 0.5).astype(int)
        if len(np.unique(yb)) > 1:
            aucs.append(roc_auc_score(yb, score[m, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def greedy(ranks, Y, W, rows, max_members, cols=None):
    """Pick members (with replacement) maximising macro AUC on `rows`."""
    chosen, cur = [], None
    best_hist = []
    for _ in range(max_members):
        best, best_i = -1.0, None
        for i, r in enumerate(ranks):
            cand = r if cur is None else (cur * len(chosen) + r) / (len(chosen) + 1)
            v = macro(cand[rows], Y[rows], W[rows], cols)
            if v > best:
                best, best_i = v, i
        if best_i is None:
            break
        chosen.append(best_i)
        cur = ranks[best_i] if cur is None else (cur * (len(chosen) - 1) + ranks[best_i]) / len(chosen)
        best_hist.append(best)
    return chosen, best_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max", type=int, default=6, help="ensemble size cap (greedy, with replacement)")
    ap.add_argument("--per-label", action="store_true",
                    help="select a separate ensemble per finding (more powerful, more overfit-prone)")
    a = ap.parse_args()

    runs, names = [], []
    for d in a.dirs:
        r = load(d)
        if r:
            runs.append(r); names.append(os.path.basename(d.rstrip("/")))
    if len(runs) < 2:
        raise SystemExit("need at least two runs with OOF files")
    uids = sorted(set.intersection(*[set(r) for r in runs]))
    print(f"{len(runs)} runs, {len(uids)} studies in common\n")
    Y = np.stack([runs[0][u][1] for u in uids])
    W = np.stack([runs[0][u][2] for u in uids])
    folds = np.array([runs[0][u][3] for u in uids])
    probs = [np.stack([r[u][0] for u in uids]) for r in runs]
    ranks = [np.stack([rankdata(p[:, j]) / len(uids) for j in range(len(LABELS))], 1) for p in probs]

    print(f"{'run':28s} {'solo macro':>11s}")
    solos = []
    for n, r in zip(names, ranks):
        v = macro(r, Y, W); solos.append(v)
        print(f"  {n:26s} {v:11.4f}")
    best_solo = int(np.argmax(solos))

    # honest: select on 4 folds, score on the held-out one
    oos_ens, oos_best = [], []
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        pick, _ = greedy(ranks, Y, W, tr, a.max)
        ens = np.mean([ranks[i] for i in pick], axis=0)
        oos_ens.append(macro(ens[te], Y[te], W[te]))
        oos_best.append(macro(ranks[best_solo][te], Y[te], W[te]))
    gain = float(np.mean(oos_ens) - np.mean(oos_best))
    print(f"\nheld-out: best single {np.mean(oos_best):.4f} -> greedy ensemble {np.mean(oos_ens):.4f}"
          f"   ({gain:+.4f})")
    if gain <= 0:
        print("  the ensemble does NOT beat the best single model out-of-sample — ship the single.")

    pick, hist = greedy(ranks, Y, W, np.ones(len(uids), bool), a.max)
    cnt = Counter(pick)
    print(f"\nfinal selection (refit on all folds), macro {hist[-1]:.4f}:")
    for i, c in cnt.most_common():
        print(f"  {c}x  {names[i]}")
    out = {"members": {names[i]: c for i, c in cnt.items()},
           "oos_gain_vs_best_single": round(gain, 5),
           "n_studies": len(uids), "runs": names}
    if a.out:
        tmp_out = a.out + ".tmp"
        with open(tmp_out, "w") as fh:
            json.dump(out, fh, indent=2)
        os.replace(tmp_out, a.out)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
