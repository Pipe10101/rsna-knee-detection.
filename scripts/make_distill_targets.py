#!/usr/bin/env python3
"""Teacher probabilities for distillation from a fold set's out-of-fold predictions.

    python3 scripts/make_distill_targets.py --oof-dir models/slotknee_t4/models --out data_subset/labels_external/teacher_t4.csv

Each study's row comes from the one fold model that held it out, so no model ever sees
its own prediction as a target.  (Across folds an indirect path exists — see
scripts/pseudo_fill.py — so use the result for final-submission training, not for A/B
arbitration.)  Use with:  train_slotknee.py --distill-targets <out.csv> --distill-weight 0.5
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.llm_labels import LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--oof-dir", nargs="+", required=True, help="one or more dirs with oof_fold_*.npz (averaged in probability space)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
acc = {}
for d in a.oof_dir:
    for p in sorted(glob.glob(os.path.join(d, "oof_fold_*.npz"))):
        z = np.load(p, allow_pickle=True)
        for uid, lg in zip(z["uids"], z["logits"]):
            acc.setdefault(str(uid), []).append(1.0 / (1.0 + np.exp(-np.asarray(lg, dtype=np.float64))))
rows = [{"StudyInstanceUID": u, **{l: float(np.mean([v[i] for v in vs])) for i, l in enumerate(LABELS)}} for u, vs in acc.items()]
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(df)} studies from {len(a.oof_dir)} fold set(s); mean prob per label:",
      {l: round(float(df[l].mean()), 3) for l in LABELS[:4]}, "...")
