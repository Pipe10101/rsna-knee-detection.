#!/usr/bin/env bash
# DIVERSITY arms — the ensemble lever, which is the largest one left (2026-08-25).
#
#   bash scripts/queue_diversity_arms.sh <cache_dir> <out_root> [epochs] [folds...]
#
# WHY: scripts/select_ensemble.py, run over the five reg4-family runs, picked ONE model five
# times out of five and gained +0.0000 out-of-sample over the best single. Those runs are the
# same architecture on the same layout (seed correlation 0.953), so there is nothing to
# ensemble. The +0.015-0.025 attributed to "diverse ensembling" is only real if the members
# are genuinely different MODELS, not different knobs.
#
# These arms exist to CREATE that diversity, then `select_ensemble.py` decides honestly
# whether it bought anything. Judge each arm two ways: its solo gate (does it stand alone?)
# and its ensemble contribution (does the selector pick it alongside r4distill?). An arm can
# fail the first and still earn its place by the second — that is the entire point.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE="${1:?cache dir}"; OUT="${2:?out root}"; EPOCHS="${3:-8}"; shift 3 || true
FOLDS=("${@:-0 1}"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1)
LABELS="data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv"
WFROM="data_subset/labels_external/stevenleehans/llm_labels_v2.csv"
REG4="vit_small_patch14_reg4_dinov2.lvd142m"
TEACHER="data_subset/labels_external/teacher_g10_oof.csv"

run_arm () {
    local NAME="$1" SEED="$2"; shift 2
    local DIR="$OUT/${NAME}_s${SEED}"
    if [ -f "$DIR/.arm_done" ]; then echo "== $NAME s$SEED: done, skip"; return; fi
    echo "== $(date '+%H:%M') $NAME seed $SEED -> $DIR"
    python3 scripts/train_slotknee.py --cache "$CACHE" --data-dir data_subset \
        --labels "$LABELS" --weights-from "$WFROM" \
        --folds ${FOLDS[@]} --epochs "$EPOCHS" --bs 8 --seed "$SEED" \
        --out "$DIR" "$@"
    touch "$DIR/.arm_done"
}

# A DIFFERENT BACKBONE. DINOv3 is a different pretraining corpus and a different patch size
# (16 vs 14: 201 tokens/img against DINOv2's 257, so ~23% less encoder time as a bonus).
# Verified end to end on CPU: trains, checkpoints, and round-trips through inference.
# Licence cleared for a final submission (POLICY 4); its text ships with the weights.
run_arm dv3      42 --backbone vit_small_patch16_dinov3.lvd1689m \
                    --pretrained-path kaggle/slotknee_code/weights
run_arm dv3dist  42 --backbone vit_small_patch16_dinov3.lvd1689m \
                    --pretrained-path kaggle/slotknee_code/weights \
                    --distill-targets "$TEACHER" --distill-weight 0.5

# ⭐ A DIFFERENT FEATURE EXTRACTION — the most promising structural idea on the board.
# Every image's 256 patch tokens are currently squashed to CLS + MEAN before anything else
# runs. A meniscus tear or MCL sprain occupies ~3-8 patches, so its evidence is attenuated
# ~30-80x AT THE ENCODER OUTPUT, before slots, attention or MIL ever see it. That is very
# likely why MIL pooling over (slot, anchor) tokens REGRESSED (-0.0176): it fought the second
# dilution while the first had already destroyed the signal. cls_mean_max keeps a peak
# statistic alongside the mean so focal evidence survives; cls_mean_topk is the softer,
# less noise-sensitive version. Costs +0.1M head params and ZERO extra encoder time, so it is
# free on the efficiency track. Targets exactly the weak findings: LatMen 0.748, MCL 0.778.
run_arm r4max     42 --backbone "$REG4" --pool cls_mean_max \
                     --distill-targets "$TEACHER" --distill-weight 0.5
run_arm r4topk    42 --backbone "$REG4" --pool cls_mean_topk \
                     --distill-targets "$TEACHER" --distill-weight 0.5

# A DIFFERENT READ-OUT. Cosine attention bounds the logits so only tau can sharpen them;
# even if it does not beat reg4 solo, a model that aggregates differently is exactly the kind
# of member an ensemble can use.
run_arm r4cosdist 42 --backbone "$REG4" --attn-cosine --attn-tau-init 10.0 \
                     --distill-targets "$TEACHER" --distill-weight 0.5

# A DIFFERENT OBJECTIVE. AUCM optimises the competition metric directly in the final epochs,
# so its errors should differ in shape from a BCE model's.
run_arm r4aucmdist 42 --backbone "$REG4" --aucm-epochs 3 \
                      --distill-targets "$TEACHER" --distill-weight 0.5

echo
echo "now ask whether any of it actually helps:"
echo "  python3 scripts/select_ensemble.py $OUT/r4distill_s42 $OUT/dv3*_s42 $OUT/r4cosdist_s42 $OUT/r4aucmdist_s42 --max 6"
