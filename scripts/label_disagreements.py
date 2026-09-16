#!/usr/bin/env python3
"""Cells where the model confidently contradicts its own training label.

At OOF time the model never saw the study, so a confident contradiction of the
LLM-read label flags a probable LABEL error (or a genuinely hard case).  Wholesale
re-reading measured no gain (labels average 0.89 vs gold); repairing only these
cells targets the errors that actually mis-teach the model.

    python3 scripts/label_disagreements.py --oof models/kaggle_traing10_v5/models/slotknee_g10 \
        --labels data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv \
        --out logs/label_disagreements.csv --threshold 0.8 --top 300
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.llm_labels import LABELS, load_llm_labels

ap = argparse.ArgumentParser()
ap.add_argument("--oof", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--data-dir", default="data_subset")
ap.add_argument("--out", required=True)
ap.add_argument("--threshold", type=float, default=0.8, help="min |model_p - label| to flag")
ap.add_argument("--top", type=int, default=300)
a = ap.parse_args()

lab = load_llm_labels([a.labels]).set_index("StudyInstanceUID")
gold = set(pd.read_csv(os.path.join(a.data_dir, "train_gold.csv")).StudyInstanceUID)
reports = pd.read_csv(os.path.join(a.data_dir, "train.csv"), usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")
rows = []
for p in sorted(glob.glob(os.path.join(a.oof, "**", "oof_fold_*.npz"), recursive=True)):
    z = np.load(p, allow_pickle=True)
    for uid, lg, w in zip(z["uids"], z["logits"], z["w"]):
        uid = str(uid)
        if uid in gold or uid not in lab.index:      # gold labels are authoritative; skip
            continue
        prob = 1.0 / (1.0 + np.exp(-np.asarray(lg, float)))
        for i, l in enumerate(LABELS):
            if w[i] <= 0:
                continue
            y = float(lab.loc[uid, l])
            d = abs(prob[i] - y)
            if d >= a.threshold:
                rows.append({"StudyInstanceUID": uid, "label": l, "label_p": round(y, 3),
                             "model_p": round(float(prob[i]), 3), "gap": round(float(d), 3)})
df = pd.DataFrame(rows).sort_values("gap", ascending=False).head(a.top)
df["Report"] = [str(reports.Report.get(u, ""))[:4000] for u in df.StudyInstanceUID]
df.to_csv(a.out, index=False)
print(f"flagged {len(rows)} cells >= {a.threshold}; wrote top {len(df)} to {a.out}")
print(df.label.value_counts().to_string())
