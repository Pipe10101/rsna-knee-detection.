#!/usr/bin/env bash
# Post-reg4 arm queue, ordered by MEASURED priority (2026-08-24).
#
#   bash scripts/queue_reg4_arms.sh <cache_dir> <out_root> [epochs] [folds...]
#
# Why this queue and not the tail of ablate_slotknee.sh:
#   * reg4 is ADOPTED (+0.0220 pooled, ACL +0.0617), so every new arm is measured ON the
#     reg4 backbone -- deltas against a superseded baseline answer a question we no longer ask.
#   * The remaining error is concentrated in findings visible in only 1-2 of the 60 anchors
#     (vs true gold: MCL 0.673, LatMen 0.648, MedMen 0.816, PFOA 0.788). A softmax over 60
#     tokens dilutes exactly that signal, so the two arms that attack dilution directly --
#     MIL log-sum-exp pooling and a hotter attention temperature -- lead the queue instead of
#     sitting 4th and 5th behind tokenpool/tb2/nomixer, which test nothing we still care about.
#   * distill2 is the only big head-knob positive we ever saw (+0.0127) and now has a
#     leak-free teacher, so it comes before the two brand-new losses.
#
# Arms are resumable: an arm with a .arm_done marker is skipped, so re-running is safe.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE="${1:?cache dir}"; OUT="${2:?out root}"; EPOCHS="${3:-8}"; shift 3 || true
FOLDS=("${@:-0 1}"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1)
LABELS="data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv"
WFROM="data_subset/labels_external/stevenleehans/llm_labels_v2.csv"
REG4="vit_small_patch14_reg4_dinov2.lvd142m"

run_arm () {  # name seed extra-args...
    local NAME="$1" SEED="$2"; shift 2
    local DIR="$OUT/${NAME}_s${SEED}"
    if [ -f "$DIR/.arm_done" ]; then echo "== $NAME s$SEED: done, skip"; return; fi
    echo "== $(date '+%H:%M') $NAME seed $SEED -> $DIR"
    python3 scripts/train_slotknee.py --cache "$CACHE" --data-dir data_subset \
        --labels "$LABELS" --weights-from "$WFROM" --backbone "$REG4" \
        --folds ${FOLDS[@]} --epochs "$EPOCHS" --bs 8 --seed "$SEED" \
        --out "$DIR" "$@"
    touch "$DIR/.arm_done"
}

# 1. The new reference: reg4 at a second seed gives the post-adopt noise floor.  Every verdict
#    below is gated against reg4_s42 (already on disk) with THIS pair's spread as the floor.
run_arm reg4 1337

# 2-3. Dilution: a finding on 1-2 of 60 tokens is averaged away by a soft attention.
run_arm r4mil   42 --mil-pool lse          # instance head + log-sum-exp over tokens
run_arm r4tau3  42 --attn-tau-init 3.0     # sharper per-label attention

# 4. Self-distillation with the leak-free teacher (old, leaked version scored +0.0127).
run_arm r4distill 42 --distill-targets data_subset/labels_external/teacher_g10_oof.csv \
                     --distill-weight 0.5

# 5-6. New objectives (implemented 2026-08-24, both default-off elsewhere).
run_arm r4aucm  42 --aucm-epochs 3         # optimise the competition metric in the last epochs
run_arm r4sam   42 --sam-rho 0.05          # sharpness-aware; evidence is strongest under label noise

# 7. Sequence structure over the anchors, then the proposed augmentations.
run_arm r4seqmix 42 --seq-mix gru
run_arm r4cutrot 42 --cutout --rot-aug

# LAST — model-repaired labels (scripts/refine_labels.py), DEMOTED 2026-08-24 19:50.
# Built because the model outscores its own targets against gold on Fracture/Contusion/Effusion.
# Then the label scout showed gold itself is ~11% wrong: two INDEPENDENT extractors agree with
# each other at 0.999 macro yet both score ~0.89 vs gold, and v4 sides with the scout against
# gold in 74 of 76 contradicted cells.  So the alphas were fitted against a corrupt reference
# and this arm is no longer the confident bet it looked like an hour ago.
# It still has ONE mechanism that stands on its own: v4 leaves 43% of studies tied at exactly
# 0.25 for Fracture and 48% tied for Baker's — unrankable blocks that blending resolves.
# ASYMMETRIC GATE: an ADOPT here is meaningful; a REGRESS is AMBIGUOUS, because the arm is
# scored against the very v4 labels it deliberately moves away from.  Runs only if time remains.
run_arm r4refined 42 --labels data_subset/labels_external/llm_labels_v6_refined.csv

python3 - "$OUT" <<'PY'
import glob, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
root = sys.argv[1]
def pooled(d):
    lg, y, w = [], [], []
    for p in sorted(glob.glob(os.path.join(d, "oof_fold_*.npz"))):
        z = np.load(p, allow_pickle=True)
        lg.append(z["logits"]); y.append(z["y"]); w.append(z["w"])
    if not lg: return None
    lg, y, w = np.concatenate(lg), np.concatenate(y), np.concatenate(w)
    a = []
    for i in range(y.shape[1]):
        m = w[:, i] > 0; yt = (y[m, i] > 0.5).astype(int)
        if len(np.unique(yt)) > 1: a.append(roc_auc_score(yt, lg[m, i]))
    return float(np.mean(a))
ref = pooled(os.path.join(root, "reg4_s42"))
ref2 = pooled(os.path.join(root, "reg4_s1337"))
noise = abs(ref - ref2) if (ref and ref2) else None
print(f"\nreference reg4_s42 = {ref}")
if noise: print(f"reg4 seed noise floor = {noise:.4f} (gate: 2x = {2*noise:.4f})")
for d in sorted(glob.glob(os.path.join(root, "r4*_s*"))):
    v = pooled(d)
    if v and ref:
        tag = "" if not noise else ("ADOPT" if v-ref > 2*noise else "REGRESS" if v-ref < -2*noise else "NULL")
        print(f"{os.path.basename(d):18s} {v:.4f}  {v-ref:+.4f}  {tag}")
PY
