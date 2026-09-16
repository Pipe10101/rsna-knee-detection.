#!/usr/bin/env python3
"""Repair the weak LLM labels using the model's own out-of-fold predictions.

    python3 scripts/refine_labels.py \
        --labels data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv \
        --oof models/kaggle_traing10_v5/models/slotknee_g10 \
        --out data_subset/labels_external/llm_labels_v6_refined.csv

WHY.  Scored against the 58 gold studies, the LLM training labels are wildly uneven: ACL 0.987
and MCL 0.968 are near-perfect, while Fracture 0.793, Contusion 0.860 and Effusion 0.877 are
weak — and on exactly those findings the MODEL beats its own targets (Fracture +0.110,
Effusion +0.086, Contusion +0.073).  Where the student outscores the teacher, the teacher
should be corrected.  Per label this writes

    refined_j = (1 - alpha_j) * llm_j + alpha_j * model_oof_j

with alpha_j chosen per label, never globally: 0 keeps the LLM label untouched, 1 replaces it.

HONESTY.  alpha is selected by cross-validation INSIDE the gold set (fit on 4/5 of the gold
studies, score on the held-out 1/5) and kept only where that out-of-sample score beats the raw
LLM label by a margin.  Gold is 58 studies, so alpha is quantised to 0.1 and clamped by
--max-alpha to keep it from chasing noise.  Lateral OA and Synovitis are excluded by default
(--untrusted): those gold cells have verified errors, so "improving" against them is not
evidence of anything.  The 58 gold rows themselves are passed through unchanged — they are
ground truth and already train at 8x weight.

GATING.  A model trained on these labels must be scored against the ORIGINAL v4-derived
targets (or gold), never against the refined ones: a model graded on labels partly built from
its own predictions agrees with itself for free.  This is the same trap the `relabel` arm hit.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.llm_labels import ID_COL, LABELS, load_llm_labels

UNTRUSTED_DEFAULT = ["Lateral OA", "Synovitis"]


def load_oof_probs(d):
    """{uid: prob vector} from a run directory's oof_fold_*.npz files."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "oof_fold_*.npz"), recursive=True)):
        z = np.load(p, allow_pickle=True)
        for uid, lg in zip(z["uids"], z["logits"]):
            out[str(uid)] = 1.0 / (1.0 + np.exp(-np.asarray(lg, dtype=float)))
    return out


def choose_alpha(y, llm, mdl, folds, alphas, min_gain, max_alpha):
    """Cross-validated alpha for one label: fit on 4/5 of gold, score on the held-out 1/5."""
    ok = np.isfinite(y) & np.isfinite(llm) & np.isfinite(mdl)
    yb = (y > 0.5).astype(int)
    gains, picks = [], []
    for k in sorted(set(folds)):
        tr = ok & (folds != k)
        te = ok & (folds == k)
        if len(np.unique(yb[tr])) < 2 or len(np.unique(yb[te])) < 2:
            continue
        best = max(alphas, key=lambda a: roc_auc_score(yb[tr], (1 - a) * llm[tr] + a * mdl[tr]))
        gains.append(roc_auc_score(yb[te], (1 - best) * llm[te] + best * mdl[te])
                     - roc_auc_score(yb[te], llm[te]))
        picks.append(best)
    if not gains or float(np.mean(gains)) < min_gain:
        return 0.0, (float(np.mean(gains)) if gains else float("nan"))
    return float(min(np.median(picks), max_alpha)), float(np.mean(gains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--oof", required=True, help="run dir holding oof_fold_*.npz")
    ap.add_argument("--gold", default="data_subset/train_gold.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-alpha", type=float, default=0.8,
                    help="never replace a label outright; 0.8 keeps a fifth of the LLM signal")
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="required held-out gain over the raw label before any blending")
    ap.add_argument("--untrusted", nargs="*", default=UNTRUSTED_DEFAULT,
                    help="labels whose gold cells are known-unreliable; never blended")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    llm_df = load_llm_labels([a.labels])
    llm_df[ID_COL] = llm_df[ID_COL].astype(str)
    probs = load_oof_probs(a.oof)
    gold = pd.read_csv(a.gold)
    gold[ID_COL] = gold[ID_COL].astype(str)
    gold = gold[gold[ID_COL].isin(probs)].reset_index(drop=True)
    guids = gold[ID_COL].tolist()
    rng = np.random.default_rng(a.seed)
    folds = rng.permutation(len(guids)) % 5
    alphas = np.round(np.arange(0.0, 1.01, 0.1), 2)

    print(f"gold studies with an out-of-fold prediction: {len(guids)}")
    print(f"{'label':18s} {'alpha':>6s} {'held-out gain':>14s}   note")
    chosen = {}
    for j, lab in enumerate(LABELS):
        if lab in a.untrusted or lab not in gold.columns:
            chosen[lab] = 0.0
            print(f"{lab:18s} {0.0:6.1f} {'-':>14s}   {'gold unreliable' if lab in a.untrusted else 'not in gold'}")
            continue
        y = pd.to_numeric(gold[lab], errors="coerce").to_numpy(dtype=float)
        l = llm_df.set_index(ID_COL).reindex(guids)[lab].to_numpy(dtype=float)
        m = np.array([probs[u][j] for u in guids], dtype=float)
        alpha, gain = choose_alpha(y, l, m, folds, alphas, a.min_gain, a.max_alpha)
        chosen[lab] = alpha
        note = "blended" if alpha > 0 else "kept (label already better)"
        print(f"{lab:18s} {alpha:6.1f} {gain:+14.4f}   {note}")

    out = llm_df[[ID_COL] + [c for c in LABELS if c in llm_df.columns]].copy()
    gold_uids = set(guids)
    n_blend = 0
    for j, lab in enumerate(LABELS):
        if lab not in out.columns or chosen[lab] <= 0:
            continue
        al = chosen[lab]
        vals = out[lab].to_numpy(dtype=float).copy()
        for i, uid in enumerate(out[ID_COL]):
            if uid in gold_uids or uid not in probs:
                continue                      # gold rows are ground truth: never overwrite
            if np.isfinite(vals[i]):
                vals[i] = (1 - al) * vals[i] + al * probs[uid][j]
                n_blend += 1
        out[lab] = vals
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    blended = [l for l in LABELS if chosen.get(l, 0) > 0]
    print(f"\nwrote {a.out}: {len(out)} studies, {n_blend} cells blended across "
          f"{len(blended)} labels {blended}")
    print("GATE REMINDER: score a model trained on this against the ORIGINAL v4 targets or gold "
          "— never against these refined labels.")


if __name__ == "__main__":
    main()
