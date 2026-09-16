#!/usr/bin/env bash
# ⛔ CLOSED 2026-08-25 — DO NOT RUN. Kept only as the record of a settled question.
#
# The per-finding attention collapses to a uniform mean (entropy 0.9994) and EVERY attempt to
# sharpen it failed on our own data: MIL pooling -0.0176, temperature -0.0010, max-pool -0.0104,
# top-k -0.0115, and an inference-time temperature sweep degraded monotonically. A 2026
# multi-dataset MIL benchmark (arXiv 2604.26807) then explained why: simple MEAN POOLING with no
# learnable attention matches or beats attention-MIL and 3D CNNs on 4 of 6 tasks while training
# 25x faster. Our head did not fail — gradient descent found the field's strong baseline.
#
# Spend arms on inputs (bagging, zoom, plane specialists) and the encoder (SSL), not here.
# Attention-collapse arms — the highest-priority experiments as of 2026-08-24 20:40.
#
#   bash scripts/queue_attention_arms.sh <cache_dir> <out_root> [epochs] [folds...]
#
# THE DIAGNOSIS these attack (docs/research_improvements_20260824.md, "THE DIAGNOSIS"):
# the trained per-finding attention is UNIFORM — entropy 0.9994, learned tau 0.955-1.000 from
# a 1.0 init, anchor spread 0.0085. The head that is supposed to let each finding choose its
# sequence and slice is mean-pooling instead, and where it does tilt it tilts to the wrong
# plane (ACL and MCL both favour AXIAL; ACL is sagittal, MCL is coronal). That single fact
# explains MCL 0.778 / LatMen 0.748, why every capacity arm was NULL, and why coverage helped
# so much (a better average, never a better selection).
#
# Per the model's own docstring, entropy is stationary at exactly-uniform attention so its
# gradient vanishes there: the TEMPERATURE is the escape, the penalty is what stops it
# re-collapsing. Hence the pairing, not either alone.
#
# Run this AFTER scripts/queue_reg4_arms.sh finishes (only one MPS training at a time).
# Arms are resumable; re-running skips anything with a .arm_done marker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE="${1:?cache dir}"; OUT="${2:?out root}"; EPOCHS="${3:-8}"; shift 3 || true
FOLDS=("${@:-0 1}"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1)
LABELS="data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv"
WFROM="data_subset/labels_external/stevenleehans/llm_labels_v2.csv"
REG4="vit_small_patch14_reg4_dinov2.lvd142m"

run_arm () {
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

# TAU CALIBRATED BY MEASUREMENT (2026-08-24 21:30), not guessed. The attention logits span
# only 0.177 across tokens, giving a max softmax probability of 0.0617 against a uniform
# 0.0556 -- barely distinguishable from flat. Scaling that spread is what tau does, and an
# inference sweep on a trained model showed how much is needed: tau x3 barely moved entropy
# (0.9948), x10 reached 0.9375, and only x30 produced genuinely selective attention (0.6614).
# So tau_init 3.0 -- the value queued before this was measured -- almost certainly UNDERSHOOTS.
run_arm r4sharp    42 --attn-tau-init 3.0  --attn-entropy-weight 0.05   # conservative, comparable to r4tau3
run_arm r4sharp10  42 --attn-tau-init 10.0 --attn-entropy-weight 0.05   # where entropy starts to move
run_arm r4sharp30  42 --attn-tau-init 30.0 --attn-entropy-weight 0.10   # genuinely selective attention

# COSINE attention (Swin-V2 style): L2-normalise query and token so the logits are bounded
# in [-1, 1] and sharpness can ONLY come from attn_tau.  This is the principled version of the
# fix: with a raw dot product the optimiser can flatten attention for free by shrinking norms,
# and the measurements say it does exactly that (logit span 0.177; learnable tau drifted DOWN
# from 1.0 to ~0.96).  Cosine removes that escape route, so tau becomes the honest control.
# Needs a high tau to matter: cos in [-1,1] means tau IS the logit scale.
run_arm r4cos      42 --attn-cosine --attn-tau-init 10.0
run_arm r4cos30    42 --attn-cosine --attn-tau-init 30.0 --attn-entropy-weight 0.05

# Penalty alone, to decompose "escape" from "stay escaped".
run_arm r4ent      42 --attn-entropy-weight 0.05

# HONEST CAVEAT — this whole family may fail, and here is the evidence against it:
#   * `attn_tau` was LEARNABLE and drifted DOWN from its 1.0 init to 0.955-1.000. The optimiser
#     actively preferred flatter attention, i.e. uniformity may be the genuine loss optimum for
#     this token set, not a pathology.
#   * Sharpening a trained model at inference degraded macro AUC monotonically
#     (0.8089 -> 0.8087 -> 0.8056 -> 0.7817 at x1/x3/x10/x30). Some of that is train/test
#     mismatch -- the head was fitted against a mean -- but none of it is encouraging.
# What keeps the family alive: the tokens ARE distinct across sequences (mean cosine 0.312, far
# from identical), so the information the attention would need to select on genuinely exists,
# and MCL/LatMen are precisely the plane-specific findings that a working selector should help.
# Judge by ENTROPY FIRST: an arm whose attn_ent stays ~1.0 never escaped and its AUC is moot.

# Watch `attn_ent` in each fold log: it is now printed every epoch and flagged above 0.98.
# An arm whose entropy stays ~1.0 has NOT escaped, whatever its AUC says.
echo
echo "attention entropy by arm (last epoch of fold 0):"
for d in "$OUT"/r4sharp_s42 "$OUT"/r4ent_s42 "$OUT"/r4sharp6_s42 "$OUT"/r4tau3_s42 "$OUT"/reg4_s42; do
    f="$d/fold_0_log.json"
    [ -f "$f" ] && python3 -c "
import json,sys
h=[e for e in json.load(open('$f')) if 'attn_entropy' in e]
print(f\"  {'$(basename $d)':16s} {h[-1]['attn_entropy']:.4f}\" if h else '  $(basename $d): not logged (pre-fix run)')"
done
