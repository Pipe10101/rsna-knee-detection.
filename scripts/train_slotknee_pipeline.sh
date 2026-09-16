#!/usr/bin/env bash
# Production pipeline for SlotKnee-S:
# 1. Pass 1 (Teacher): Trains folds on v4_blend labels (and v2 for silence-fixed weights)
# 2. Pseudo-Labeling: Uses the Teacher OOFs to fill silent cells with highly-confident predictions
# 3. Pass 2 (Student): Trains folds on the new pseudo-labeled dataset
# 4. Souping: Averages the Student folds into a single soup.pt (wrapped with fold-0
#    hparams/slot_layout so scripts/infer_slotknee.py loads it), then the OOF report.
#
# Usage:
#   bash scripts/train_slotknee_pipeline.sh [CACHE] [OUT] [EPOCHS] [BS] [FOLDS...]
# Env:
#   SEED (42), FOLD_SEED (42; keep 42 so runs stay fold-matched with the campaign's
#   ablation arms), LIMIT_MINUTES (per-pass wall-clock budget -> --limit-minutes),
#   TRAINABLE_BLOCKS (4; tb6 measured NULL: +0.0001 pooled OOF, see docs/slotknee_runbook.md),
#   PSEUDO_LABELS (where pseudo_fill.py writes its CSV),
#   EXTRA_TRAIN_ARGS (extra train_slotknee.py flags, e.g. "--amp cpu --max-studies 6").
#   DISTILL_WEIGHT (comma-separated list of 12 floats for per-label distillation weighting).
#
# Caches: 
#   The default cache is g10t1 (10 anchors, 1 slice) at 224px.
#   If testing the 336px full-coverage cache, pass CACHE="cache/slots_P336_g10t1_full".
#   If testing the Coronal Joint-Line Zoom cache, pass CACHE="cache/slots_P224_zoom_cor_joint".
#
# Fixed 2026-08-24 (audit):
#   * The old single `python3 -m src.soup --checkpoints ... --out ... --oof-report`
#     call returned early on --oof-report and NEVER wrote soup.pt; and src.soup
#     averages RAW state dicts, so the SlotKnee {state_dict, hparams, slot_layout,
#     args} checkpoints crash it. Both handled by scripts/soup_slotknee.py, called
#     twice (soup first, report second).
#   * $2 (OUT) was ignored; restored. FOLDS/--fold-seed/--limit-minutes now pass through.
#   * trainable-blocks 6 -> default 4 (tb6 measured NULL on folds 0-1).
#
# POLICY: the soup is NOT the default submission. Measured: a 5-member ViT-S T4
# ensemble costs ~0.0006 efficiency units (0.01 AUC ~ 717 s); ship a soup only if
# a soup-vs-ensemble OOF measurement puts it within noise. The efficiency-track
# entry is the distilled single student. See docs/slotknee_runbook.md and
# the campaign log (2026-08-24 01:05).
#
# Leakage note: pass-2 OOF numbers are mildly inflated (see scripts/pseudo_fill.py
# docstring) -- use this pipeline for FINAL submission models, not for A/B decisions.
#
# CROSS-ARCHITECTURE DISTILLATION:
# To achieve >0.90 AUC but retain maximum speed, Pass 1 & 2 now use a heavy
# ViT-Base backbone (Teacher). Pass 3 explicitly distills that intelligence down
# into a ViT-Small backbone (Student). Since 5 folds of ViT-Base may exceed
# Kaggle's 9h limit, you may need to run Folds 0-2 in one kernel, and Folds 3-4
# in a separate kernel, and then merge the output directories before Distillation.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE="${1:-cache/slots_P224_g10t1_full/slots_P224_g10t1}"   # adopted coverage layout
OUT="${2:-models/slotknee_prod}"
EPOCHS="${3:-8}"
BS="${4:-8}"
if [ "$#" -ge 4 ]; then shift 4; else shift "$#"; fi
FOLDS="${*:-0 1 2 3 4}"

OUT_TEACHER="${OUT}_teacher"
OUT_STUDENT="$OUT"
SEED="${SEED:-42}"
FOLD_SEED="${FOLD_SEED:-42}"
TRAINABLE_BLOCKS="${TRAINABLE_BLOCKS:-4}"
TEACHER_BACKBONE="${TEACHER_BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
STUDENT_BACKBONE="${STUDENT_BACKBONE:-vit_small_patch14_dinov2.lvd142m}"
LIMIT_MINUTES="${LIMIT_MINUTES:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"

LABELS="data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv"
WFROM="data_subset/labels_external/stevenleehans/llm_labels_v2.csv"
PSEUDO_LABELS="${PSEUDO_LABELS:-data_subset/labels_external/stevenleehans/llm_labels_v4_blend_pseudo.csv}"

LIMIT_ARGS=""
if [ -n "$LIMIT_MINUTES" ]; then
    LIMIT_ARGS="--limit-minutes $LIMIT_MINUTES"
fi

echo "=== 1. PASS 1 (Teacher): Training folds $FOLDS ==="
echo "Cache: $CACHE | Teacher out: $OUT_TEACHER | Student out: $OUT_STUDENT"
echo "Seed: $SEED | Fold seed: $FOLD_SEED | tb: $TRAINABLE_BLOCKS | Limit: ${LIMIT_MINUTES:-none}"
mkdir -p "$OUT_TEACHER"

# shellcheck disable=SC2086  # FOLDS / LIMIT_ARGS / EXTRA_TRAIN_ARGS are flag lists
python3 scripts/train_slotknee.py \
    --cache "$CACHE" \
    --data-dir data_subset \
    --labels "$LABELS" \
    --weights-from "$WFROM" \
    --folds $FOLDS \
    --epochs "$EPOCHS" \
    --bs "$BS" \
    --seed "$SEED" \
    --fold-seed "$FOLD_SEED" \
    --trainable-blocks "$TRAINABLE_BLOCKS" \
    --backbone "$TEACHER_BACKBONE" \
    --out "$OUT_TEACHER" \
    $LIMIT_ARGS $EXTRA_TRAIN_ARGS

# NOTE: no default augs here -- cutout/rot-aug are UNGATED until the cutrot arm reports;
# pass them via EXTRA_TRAIN_ARGS only after an ADOPT verdict (one delta per experiment).
echo "=== 2. Pseudo-Labeling Silent Cells ==="
python3 scripts/pseudo_fill.py \
    --oof-dir "$OUT_TEACHER" \
    --labels "$LABELS" \
    --weights-from "$WFROM" \
    --out "$PSEUDO_LABELS" \
    --pseudo-weight 0.3 \
    --margin 0.3

echo "=== 3. PASS 2 (Student): Training folds $FOLDS on Pseudo-Labels ==="
mkdir -p "$OUT_STUDENT"

# Note: --weights-from is intentionally omitted because pseudo_fill.py bakes
# weights inside the CSV (train_slotknee.py detects the y_/w_ schema and keeps them).
# shellcheck disable=SC2086
python3 scripts/train_slotknee.py \
    --cache "$CACHE" \
    --data-dir data_subset \
    --labels "$PSEUDO_LABELS" \
    --folds $FOLDS \
    --epochs "$EPOCHS" \
    --bs "$BS" \
    --seed "$SEED" \
    --fold-seed "$FOLD_SEED" \
    --trainable-blocks "$TRAINABLE_BLOCKS" \
    --backbone "$TEACHER_BACKBONE" \
    --out "$OUT_STUDENT" \
    $LIMIT_ARGS $EXTRA_TRAIN_ARGS

echo "=== 4. Souping Student folds into 1 model ==="
N_CKPT=$(ls "$OUT_STUDENT"/fold_*_best.pt 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_CKPT" -lt 2 ]; then
    echo "Error: found $N_CKPT checkpoint(s); cannot soup. Did the training complete successfully?"
    exit 1
fi

# Two calls on purpose: src.soup's --oof-report returns early, so soup and
# report must never share one invocation (that bug ate soup.pt entirely).
python3 scripts/soup_slotknee.py "$OUT_STUDENT" --out "$OUT_STUDENT/soup.pt"

echo ""
echo "=== 5. Pooled OOF report (leak-inflated: pass-2 models saw teacher pseudo-labels) ==="
python3 scripts/soup_slotknee.py "$OUT_STUDENT" --report

echo ""
echo "=== 6. PASS 3 (Efficiency Entry): Distilling Ensemble into Single Student ==="
DISTILL_CSV="data_subset/labels_external/stevenleehans/distill_targets.csv"
OUT_DISTILLED="${OUT}_distilled"

python3 scripts/make_distill_targets.py \
    --oof-dir "$OUT_STUDENT" \
    --out "$DISTILL_CSV"

mkdir -p "$OUT_DISTILLED"

# shellcheck disable=SC2086
python3 scripts/train_slotknee.py \
    --cache "$CACHE" \
    --data-dir data_subset \
    --labels "$PSEUDO_LABELS" \
    --distill-targets "$DISTILL_CSV" \
    --distill-weight "$DISTILL_WEIGHT" \
    --folds 0 \
    --epochs "$EPOCHS" \
    --bs "$BS" \
    --seed "$SEED" \
    --fold-seed "$FOLD_SEED" \
    --trainable-blocks "$TRAINABLE_BLOCKS" \
    --backbone "$STUDENT_BACKBONE" \
    --out "$OUT_DISTILLED" \
    $LIMIT_ARGS $EXTRA_TRAIN_ARGS

echo ""
echo "=== Pipeline Complete ==="
echo "Fold checkpoints (the default submission ensemble): $OUT_STUDENT/fold_*_best.pt"
echo "Souped single model (candidate ONLY if OOF-equal to the ensemble): $OUT_STUDENT/soup.pt"
echo "Distilled efficiency model (The Fast Engine): $OUT_DISTILLED/fold_0_best.pt"
