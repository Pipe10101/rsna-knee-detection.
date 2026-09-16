#!/usr/bin/env python3
"""Fill the report-silent label cells with OOF pseudo-labels (roadmap item 5, offline).

What it does
------------
~25% of (study, label) cells have weight 0 because the radiology report never
addresses the finding.  This tool fills ONLY those cells with the model's own
out-of-fold predictions, gated by a confidence margin, and emits a ready-to-train
labels CSV in the y_/w_ schema that ``llm_labels.build_targets`` accepts:

    y from --labels (v4 blend)   where the report speaks
    w from 2|p-0.5| of --weights-from (v2)
    y = sigmoid(OOF logit), w = pseudo_weight * 2|p-0.5|
                                 where the report is SILENT and |p-0.5| >= margin

Train with:  --labels <out.csv>   and NO --weights-from (weights are inside).

Leakage note (read before using in an A/B)
------------------------------------------
Each pseudo cell comes from the fold model that held that study out, so no model
ever saw its own training row's gold/derived label.  But an INDIRECT path exists:
fold-j labels -> teacher k -> pseudo cell on a fold-k row -> student j.  With
single-holdout teachers this cannot be avoided without two-fold-out teachers.
Therefore: use this file to train FINAL submission models; do not use CV runs on
it to arbitrate between two designs (the OOF number will be mildly inflated).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.llm_labels import LABELS, load_llm_labels, cell_weights, y_col, w_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", required=True, help="dir with oof_fold_*.npz")
    ap.add_argument("--labels", required=True, help="targets CSV (v4 blend)")
    ap.add_argument("--weights-from", required=True, help="silence CSV (v2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pseudo-weight", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.3,
                    help="keep a pseudo cell only if |p-0.5| >= margin")
    args = ap.parse_args()

    y_df = load_llm_labels([args.labels]).set_index("StudyInstanceUID")
    w_src = load_llm_labels([args.weights_from]).set_index("StudyInstanceUID")
    w_df = pd.DataFrame(cell_weights(w_src[LABELS].values),
                        index=w_src.index, columns=LABELS)
    # studies present in targets but absent from the silence file keep v4 weights
    missing = y_df.index.difference(w_df.index)
    if len(missing):
        w_df = pd.concat([w_df, pd.DataFrame(
            cell_weights(y_df.loc[missing, LABELS].values),
            index=missing, columns=LABELS)])

    # OOF logits, one row per study from the fold that held it out
    oof = {}
    n_folds = 0
    for k in range(10):
        p = os.path.join(args.oof_dir, f"oof_fold_{k}.npz")
        if not os.path.isfile(p):
            continue
        z = np.load(p, allow_pickle=True)
        n_folds += 1
        for uid, lg in zip(z["uids"], z["logits"]):
            oof[str(uid)] = 1.0 / (1.0 + np.exp(-np.asarray(lg, dtype=np.float64)))
    if not oof:
        raise SystemExit(f"no oof_fold_*.npz under {args.oof_dir}")

    filled = np.zeros(len(LABELS), dtype=int)
    silent = np.zeros(len(LABELS), dtype=int)
    rows = []
    for uid, y_row in y_df[LABELS].iterrows():
        w_row = w_df.loc[uid, LABELS] if uid in w_df.index else pd.Series(
            cell_weights(y_row.values), index=LABELS)
        y_out, w_out = y_row.values.astype(float), w_row.values.astype(float)
        p = oof.get(str(uid))
        for i in range(len(LABELS)):
            if w_out[i] > 0:
                continue
            silent[i] += 1
            if p is None or abs(p[i] - 0.5) < args.margin:
                continue
            y_out[i] = float(p[i])
            w_out[i] = args.pseudo_weight * 2.0 * abs(p[i] - 0.5)
            filled[i] += 1
        row = {"StudyInstanceUID": uid}
        row.update({y_col(l): y_out[i] for i, l in enumerate(LABELS)})
        row.update({w_col(l): w_out[i] for i, l in enumerate(LABELS)})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    report = {"n_studies": len(out), "n_folds": n_folds,
              "pseudo_weight": args.pseudo_weight, "margin": args.margin,
              "silent_cells": int(silent.sum()), "filled_cells": int(filled.sum()),
              "filled_by_label": {l: int(filled[i]) for i, l in enumerate(LABELS)},
              "silent_by_label": {l: int(silent[i]) for i, l in enumerate(LABELS)}}
    print(json.dumps(report, indent=2))
    with open(args.out + ".report.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
