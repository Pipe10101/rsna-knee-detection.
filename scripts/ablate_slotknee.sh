#!/usr/bin/env bash
# Roadmap items 3+4 as runnable, paired experiments (docs/slotknee_review.md protocol:
# same code revision, 2 seeds per arm, compare POOLED OOF, state the noise floor).
#
#   bash scripts/ablate_slotknee.sh <cache_dir> <out_root> [epochs] [folds...]
#
# Arms: baseline(tb4) | tb2 | tb6 | aux0.2 | P252 (only if a matching 252 cache exists
# at <cache_dir>_P252). Each arm x 2 seeds. Prints a pooled-OOF table at the end.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE="${1:?cache dir}"; OUT="${2:?out root}"; EPOCHS="${3:-8}"; shift 3 || true
FOLDS=("${@:-0 1 2 3 4}"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1 2 3 4)
LABELS="data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv"
WFROM="data_subset/labels_external/stevenleehans/llm_labels_v2.csv"

run_arm () {  # name seed extra-args...   (ARM_CACHE overrides the cache for one arm)
    local NAME="$1" SEED="$2"; shift 2
    local DIR="$OUT/${NAME}_s${SEED}"
    if [ -f "$DIR/.arm_done" ]; then echo "== $NAME s$SEED: done, skip"; return; fi
    echo "== $NAME seed $SEED -> $DIR"
    python3 scripts/train_slotknee.py --cache "${ARM_CACHE:-$CACHE}" --data-dir data_subset \
        --labels "$LABELS" --weights-from "$WFROM" \
        --folds ${FOLDS[@]} --epochs "$EPOCHS" --bs 8 --seed "$SEED" \
        --out "$DIR" "$@"
    touch "$DIR/.arm_done"
}

# baseline = script defaults (aux 0.2, slot-dropout 0.1, bce, tb4, no token pooling).
# Laptop queue after step 1 passed (coverage adopted on T4): the laptop answers what the T4
# sessions do not — the seed noise floor first, then recipe knobs on the 224 base (fast),
# then resolution on the coverage layout (336 px, 4 single-slice anchors) when its cache exists.
run_arm baseline 1337                                   # noise floor (same folds: fold seed is fixed)
for SEED in ${SEEDS:-42}; do
    run_arm baseline  "$SEED"
    if [ -d "${CACHE336:-/nonexistent}" ]; then ARM_CACHE="$CACHE336" run_arm p336 "$SEED"; fi
    run_arm distill   "$SEED" --distill-targets data_subset/labels_external/teacher_t4.csv --distill-weight 0.5
    # relabel: v4 + 11 adjudicated corrections + 380 mild-OA downgrades (main, 2026-08-24).
    # GATE CAVEAT: its OOF y differs from v4 on 391 cells -- score its logits against the
    # v4-based y (baseline npz, common uids) and gold; its own-y AUC is not the gate metric.
    run_arm relabel   "$SEED" --labels data_subset/labels_external/llm_labels_v5_repaired.csv
    run_arm ep14      "$SEED" --epochs 14
    run_arm tb12      "$SEED" --trainable-blocks 12
    run_arm lrb1e4    "$SEED" --lr-backbone 1e-4
    run_arm reg4      "$SEED" --backbone vit_small_patch14_reg4_dinov2.lvd142m
    run_arm noaux     "$SEED" --aux-weight 0
    # --- added 2026-08-24 13:20 at a driver restart between arms (legal edit window: no
    # driver process was running the script). Promoted AHEAD of the tokenpool/tb2/nomixer/
    # tau3/mil knobs on evidence: POLICY 3 (geometry over capacity) plus the mean-pool-parity
    # benchmark make those five low-yield, while distill2 is the campaign's top-priority arm
    # (the leaked distill signalled +0.0127) and aucm targets the competition metric directly.
    # distill2 teacher = per-fold g10 OOF (leak-free by construction).
    run_arm distill2  "$SEED" --distill-targets data_subset/labels_external/teacher_g10_oof.csv --distill-weight 0.5
    run_arm aucm      "$SEED" --aucm-epochs 3
    run_arm sam       "$SEED" --sam-rho 0.05
    run_arm tokenpool "$SEED" --token-pooling
    run_arm tb2       "$SEED" --trainable-blocks 2
    run_arm nomixer   "$SEED" --mixer-layers 0
    run_arm tau3      "$SEED" --attn-tau-init 3.0
    run_arm mil       "$SEED" --mil-pool lse
    run_arm seqmix    "$SEED" --seq-mix gru
    run_arm rankloss  "$SEED" --rank-loss-weight 0.3
    run_arm cutrot    "$SEED" --cutout --rot-aug
    if [ -d "${CACHEG10:-/nonexistent}" ]; then ARM_CACHE="$CACHEG10" run_arm g10sub5 "$SEED" --group-subsample 5; fi
done

python3 - "$OUT" <<'PY'
import glob, json, os, sys
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
    aucs = []
    for i in range(y.shape[1]):
        m = w[:, i] > 0; yt = (y[m, i] > 0.5).astype(int)
        if len(np.unique(yt)) > 1: aucs.append(roc_auc_score(yt, lg[m, i]))
    return float(np.mean(aucs)), len(y)
arms = {}
for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
    r = pooled(d)
    if r: arms.setdefault(os.path.basename(d).rsplit("_s", 1)[0], []).append(r[0])
base = arms.get("baseline")
noise = abs(base[0] - base[1]) if base and len(base) > 1 else None
print(f"\n{'arm':10s} {'seed-mean':>9s} {'seeds':>18s}   vs baseline")
for a, v in sorted(arms.items()):
    d = "" if not base else f"{np.mean(v) - np.mean(base):+.4f}"
    print(f"{a:10s} {np.mean(v):9.4f} {str([round(x,4) for x in v]):>18s}   {d}")
if noise is not None:
    print(f"\nseed-to-seed noise floor (baseline): {noise:.4f} — "
          "trust only deltas > 2x this.")
PY
