#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  RSNA Knee Abnormality Detection — 5-hour Kaggle submission pipeline
#
#  Two backbones x 5 folds, each fold collapsed by Stochastic Weight Averaging,
#  then each backbone's folds collapsed into ONE set of weights by a model soup.
#  Inference ships the soup when it provably agrees with the full ensemble, so
#  the usual K-model inference bill is paid once instead of K times.
#
#  The 5-hour limit is ENFORCED, not estimated: every stage gets a wall-clock
#  budget, every fold is resumable, and a valid submission.csv exists from the
#  first minute onward.
#
#  Usage:  bash scripts/train_kaggle_5h.sh [DATA_DIR]
#  Env:    TOTAL_BUDGET_MIN   total wall clock (default 300 = 5 h)
#          BACKBONE_A_WEIGHTS offline timm weights for backbone A (Kaggle is
#          BACKBONE_B_WEIGHTS offline — timm CANNOT download at runtime)
#          FOLDS              fold list (default "0 1 2 3 4")
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail          # NOT -e: a failed fold must not kill the submission
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_DIR="${1:-data_subset}"
MODELS_DIR="models"
FOLDS="${FOLDS:-0 1 2 3 4}"
TOTAL_BUDGET_MIN="${TOTAL_BUDGET_MIN:-300}"

# ── Budget split ────────────────────────────────────────────────────────────
# Inference is the only stage whose cost scales with the *hidden* test set, so
# it gets the largest single reservation and its own internal guard.  The
# reserve absorbs Kaggle's slow startup, DICOM decode and the final CSV write.
RESERVE_MIN=25
INFER_MIN=75
TRAIN_MIN=$(( TOTAL_BUDGET_MIN - RESERVE_MIN - INFER_MIN ))
PER_BACKBONE_MIN=$(( TRAIN_MIN / 2 ))
N_FOLDS_COUNT=$(echo "$FOLDS" | wc -w | tr -d ' ')
PER_FOLD_MIN=$(( PER_BACKBONE_MIN / N_FOLDS_COUNT ))

# ── Backbones ───────────────────────────────────────────────────────────────
# Sized to the data, not to the GPU.  There are 58 labelled studies; a fold
# trains on ~46.  EfficientNetV2-M (54M) and ConvNeXt-Base (88M) have far more
# capacity than 46 studies can constrain, and they cost 3-4x the time of these.
# Diversity comes from architecture AND resolution, which is free.
BACKBONE_A="tf_efficientnetv2_s.in21k_ft_in1k"   # 21M, BatchNorm, 224px
SIZE_A=224
BACKBONE_B="convnext_tiny.fb_in22k_ft_in1k"      # 28M, LayerNorm, 288px
SIZE_B=288

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   RSNA Knee — 5h Pipeline (SWA folds → model soup → 1 model)     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Data          : $DATA_DIR"
echo "Total budget  : ${TOTAL_BUDGET_MIN} min"
echo "  training    : ${TRAIN_MIN} min (${PER_BACKBONE_MIN}/backbone, ${PER_FOLD_MIN}/fold)"
echo "  inference   : ${INFER_MIN} min"
echo "  reserve     : ${RESERVE_MIN} min"
echo "Backbone A    : ${BACKBONE_A} @ ${SIZE_A}px"
echo "Backbone B    : ${BACKBONE_B} @ ${SIZE_B}px"
echo "Folds         : ${FOLDS}"
echo ""

# ── Stage 0: a scoreable submission before anything can fail ────────────────
# If the notebook dies at any later point, this file is what gets scored.
python3 - "$DATA_DIR" <<'PY'
import os, sys, shutil
d = sys.argv[1]
src = os.path.join(d, "sample_submission.csv")
if os.path.exists(src) and not os.path.exists("submission.csv"):
    shutil.copy(src, "submission.csv")
    print("Stage 0: seeded submission.csv from sample_submission.csv")
else:
    print("Stage 0: submission.csv already present or no template found")
PY
echo ""

# ── Stage 0.5: supervision from the radiology reports ───────────────────────
# 58 studies carry expert labels; all 649 with DICOMs carry a report. src/labels.py
# turns that text into CALIBRATED SOFT targets across nine
# languages -- en/es/tr/hr/el/bg/de/nl/fr -- lifting supervision from 46 gold
# studies to 637 training rows (4661 supervised cells, all soft).
#
# Measured, effnet_b0 / 4 epochs / seed 42, mean val_auc over folds 0-2:
#   gold only 0.5028 (chance)  ->  report labels 0.5623
# This is the single largest lever in the pipeline, larger than any backbone
# choice, so it is on by default here.
#
# NOTE: src/report_labels.py is an earlier, weaker rules engine (2 languages,
# hard 0/1 labels, mean 0.5577). src/labels.py supersedes it; it is kept only
# because `--evaluate` prints a readable per-label precision/recall table.
LABELS_ARGS="use_report_labels=true report_label_weight=0.35"
echo "Report labels: src/labels.py (9 languages, soft targets, weight 0.35)"
echo "  gold cells stay at weight 1.0; validation is gold-only, always."
echo ""

# ── Training ────────────────────────────────────────────────────────────────
# One process per fold, each with its own wall-clock budget, so a slow fold can
# only ever consume its own slice.  train.py checkpoints every epoch and resumes
# from fold_N_last.pt, so a fold cut off by its budget is not lost work.
train_backbone () {
    local NAME="$1" BACKBONE="$2" SIZE="$3" LR="$4" BS="$5" ACC="$6" WEIGHTS="$7"
    local OUT="${MODELS_DIR}/${NAME}"
    mkdir -p "$OUT"

    echo "══════════════════════════════════════════════════════════════════"
    echo "TRAIN ${NAME}: ${BACKBONE} @ ${SIZE}px, ${PER_FOLD_MIN} min/fold"
    echo "══════════════════════════════════════════════════════════════════"

    for FOLD in $FOLDS; do
        echo "── ${NAME} fold ${FOLD} ──"
        # Refit the report-label calibration on every gold fold EXCEPT this one,
        # so the prior and the per-verdict reliability table carry no trace of
        # the studies about to be validated.
        CAL_FOLDS=$(python3 -c "print(','.join(str(i) for i in range(5) if i != $FOLD))")
        python3 -m src.train --folds "$FOLD" \
            --set \
            data_dir="$DATA_DIR" \
            backbone="$BACKBONE" \
            backbone_weights="$WEIGHTS" \
            image_size="$SIZE" \
            n_folds=5 \
            epochs=12 \
            lr="$LR" \
            batch_size="$BS" \
            grad_accum_steps="$ACC" \
            loss=asl \
            selection_metric=auto \
            weight_decay=0.05 \
            drop_path_rate=0.2 \
            llrd_factor=0.2 \
            use_swa=true \
            swa_start_frac=0.6 \
            use_ema=false \
            val_tta=true \
            early_stopping_patience=4 \
            min_epochs=4 \
            time_budget_min="$PER_FOLD_MIN" \
            models_dir="$OUT" \
            $LABELS_ARGS \
            report_label_calibrate_on_folds="$CAL_FOLDS" \
            || echo "WARNING: ${NAME} fold ${FOLD} exited non-zero — continuing."
    done
}

# use_ema=false is deliberate: EMA keeps a second full copy of the weights in
# VRAM for the entire fold, while SWA keeps its running average on the CPU.
# Both are weight averages; running only SWA frees ~1 model of VRAM per fold
# and lets the batch size stay where it is.
train_backbone "effnetv2s" "$BACKBONE_A" "$SIZE_A" 3e-4 8 2 "${BACKBONE_A_WEIGHTS:-}"
train_backbone "convnext_t" "$BACKBONE_B" "$SIZE_B" 2e-4 4 4 "${BACKBONE_B_WEIGHTS:-}"

# ── Collect whatever actually finished ──────────────────────────────────────
# The SWA checkpoint only exists when it BEAT the val-selected checkpoint on the
# held-out fold (train.py deletes it otherwise), so preferring it is safe here.
collect () {
    local DIR="$1"; local OUT=""
    for FOLD in $FOLDS; do
        if   [ -f "${DIR}/fold_${FOLD}_swa.pt" ];  then OUT="$OUT ${DIR}/fold_${FOLD}_swa.pt"
        elif [ -f "${DIR}/fold_${FOLD}_best.pt" ]; then OUT="$OUT ${DIR}/fold_${FOLD}_best.pt"
        fi
    done
    echo "$OUT"
}

CK_A=$(collect "${MODELS_DIR}/effnetv2s")
CK_B=$(collect "${MODELS_DIR}/convnext_t")

# Pooled out-of-fold report per backbone. This is the ONLY honest number the run
# produces about model quality — every row was predicted by the one fold model
# that never trained on it. Compare configurations on this, never on a mean of
# per-fold AUCs: a label with too few positives in a fold is uncomputable there
# and silently drops out of that fold's macro average.
for D in "${MODELS_DIR}/effnetv2s" "${MODELS_DIR}/convnext_t"; do
    if ls "$D"/oof_fold_*.npz >/dev/null 2>&1; then
        echo ""
        echo "── Out-of-fold report: $(basename "$D") ──"
        python3 -m src.soup --oof-report --models-dir "$D" || true
    fi
done
echo ""
echo "Backbone A checkpoints:$CK_A"
echo "Backbone B checkpoints:$CK_B"

# ── Soup: K fold models → 1 model, per backbone ─────────────────────────────
SOUP_A="${MODELS_DIR}/effnetv2s/soup.pt"
SOUP_B="${MODELS_DIR}/convnext_t/soup.pt"
SOUP_ARGS=""
SOUP_BB=""
SOUP_SZ=""

if [ -n "$CK_A" ]; then
    echo ""
    echo "── Souping backbone A ──"
    if python3 -m src.soup --checkpoints $CK_A --out "$SOUP_A" \
            --models-dir "${MODELS_DIR}/effnetv2s"; then
        SOUP_ARGS="$SOUP_ARGS $SOUP_A"; SOUP_BB="$SOUP_BB $BACKBONE_A"; SOUP_SZ="$SOUP_SZ $SIZE_A"
    else
        echo "Backbone A soup rejected — its folds will be logit-ensembled instead."
    fi
fi
if [ -n "$CK_B" ]; then
    echo ""
    echo "── Souping backbone B ──"
    if python3 -m src.soup --checkpoints $CK_B --out "$SOUP_B" \
            --models-dir "${MODELS_DIR}/convnext_t"; then
        SOUP_ARGS="$SOUP_ARGS $SOUP_B"; SOUP_BB="$SOUP_BB $BACKBONE_B"; SOUP_SZ="$SOUP_SZ $SIZE_B"
    else
        echo "Backbone B soup rejected — its folds will be logit-ensembled instead."
    fi
fi

# ── Inference ───────────────────────────────────────────────────────────────
# --soup is the cheap path, --checkpoints is the reference. infer.py measures
# their agreement on real test studies and ships the soup only if the two would
# rank the test set the same way; otherwise it falls back to the full ensemble.
ALL_CK="$CK_A $CK_B"
ALL_BB=""
ALL_SZ=""
for _ in $CK_A; do ALL_BB="$ALL_BB $BACKBONE_A"; ALL_SZ="$ALL_SZ $SIZE_A"; done
for _ in $CK_B; do ALL_BB="$ALL_BB $BACKBONE_B"; ALL_SZ="$ALL_SZ $SIZE_B"; done

if [ -z "$(echo "$ALL_CK" | tr -d ' ')" ]; then
    echo "No checkpoints trained. The Stage 0 submission stands."
    exit 0
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "INFERENCE (budget ${INFER_MIN} min)"
echo "══════════════════════════════════════════════════════════════════"

python3 -m src.infer \
    ${SOUP_ARGS:+--soup $SOUP_ARGS --soup-backbones $SOUP_BB --soup-sizes $SOUP_SZ} \
    --checkpoints $ALL_CK \
    --backbones $ALL_BB \
    --image-sizes $ALL_SZ \
    --output submission.csv \
    --time-budget-min "$INFER_MIN" \
    --agreement-sample 96 \
    --agreement-threshold 0.98 \
    --set \
    data_dir="$DATA_DIR" \
    tta_n=5

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     submission.csv is ready                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
head -2 submission.csv
