#!/usr/bin/env bash
# TOP-PRIORITY arms derived from the competition forum (2026-08-25), not from our own guesses.
#
#   bash scripts/queue_forum_arms.sh <cache_dir> <out_root> [epochs] [folds...]
#   (use the g10t1 COVERAGE cache: bagging needs G>K to have anything to sample from)
#
# WHY THESE TWO, AND WHY THEY OUTRANK EVERYTHING ELSE WE HAD QUEUED.
# Reading the competition's own discussion threads repositioned us badly and usefully:
# teams report 0.929-0.947 at 224 px (our resolution) while we sit at 0.866. Two differences
# stand out, and both were mis-ranked by our internal analysis:
#
#   1. FIXED vs SAMPLED SLICES. We feed the SAME 60 images every epoch. The 18th-place team
#      "randomly samples a bag of 32 slices per study across all studies during training".
#      That buys augmentation, exposure to every cached slice over epochs, and an
#      ensemble-like average. Our anchor-COUNT probe showed saturation; anchor-SAMPLING is a
#      different axis we never tested. --anchor-bag implements it properly (it SELECTS K
#      anchors; --group-subsample merely zeroes them, so the encoder still burns compute on
#      blanks and the head still sees dead tokens).
#
#   2. LABEL QUALITY, with the polarity we had BACKWARDS. The competition's targets are EXPERT
#      ANNOTATIONS (~19 skeletal radiologists — see the Acknowledgements); the reports are a
#      second modality, not the label source. So our re-extraction "contradicting gold in 11%
#      of cells" was the REPORT disagreeing with the EXPERT, not gold being wrong. The 4th-place
#      competitor (also top of the Efficiency LB) says he "spent a lot of time trying to figure
#      out how to work around the low quality labels". Our 58 expert-labelled studies are the
#      only clean signal we have, and they currently train at only 8x weight.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE="${1:?cache dir}"; OUT="${2:?out root}"; EPOCHS="${3:-8}"; shift 3 || true
FOLDS=("${@:-0 1}"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1)
LABELS="data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv"
WFROM="data_subset/labels_external/stevenleehans/llm_labels_v2.csv"
REG4="vit_small_patch14_reg4_dinov2.lvd142m"
TEACHER="data_subset/labels_external/teacher_g10_oof.csv"

run_arm_ep () {   # like run_arm, but the 3rd arg overrides EPOCHS for this arm only
    local NAME="$1" SEED="$2" EP="$3"; shift 3
    local DIR="$OUT/${NAME}_s${SEED}"
    if [ -f "$DIR/.arm_done" ]; then echo "== $NAME s$SEED: done, skip"; return; fi
    echo "== $(date '+%H:%M') $NAME seed $SEED epochs $EP -> $DIR"
    python3 scripts/train_slotknee.py --cache "$CACHE" --data-dir data_subset \
        --labels "$LABELS" --weights-from "$WFROM" --backbone "$REG4" \
        --distill-targets "$TEACHER" --distill-weight 0.5 \
        --folds ${FOLDS[@]} --epochs "$EP" --bs 8 --seed "$SEED" \
        --out "$DIR" "$@"
    touch "$DIR/.arm_done"
}

run_arm () {
    local NAME="$1" SEED="$2"; shift 2
    local DIR="$OUT/${NAME}_s${SEED}"
    if [ -f "$DIR/.arm_done" ]; then echo "== $NAME s$SEED: done, skip"; return; fi
    echo "== $(date '+%H:%M') $NAME seed $SEED -> $DIR"
    python3 scripts/train_slotknee.py --cache "$CACHE" --data-dir data_subset \
        --labels "$LABELS" --weights-from "$WFROM" --backbone "$REG4" \
        --distill-targets "$TEACHER" --distill-weight 0.5 \
        --folds ${FOLDS[@]} --epochs "$EPOCHS" --bs 8 --seed "$SEED" \
        --out "$DIR" "$@"
    touch "$DIR/.arm_done"
}

# 1. Random slice bagging — the architectural difference from the stronger public solutions.
run_arm bag6      42 --anchor-bag 6      # sample 6 of 10 anchors fresh each epoch
run_arm bag8      42 --anchor-bag 8      # gentler; 8 of 10

# 2. Trust the EXPERT labels more. 58 studies, currently 8x; the reports are the noisy proxy.
run_arm gold32    42 --gold-weight 32

# 3. Both, if either gates.
run_arm bag8gold32 42 --anchor-bag 8 --gold-weight 32

echo
echo "gate against models/ablate_full/r4distill_s42 (same backbone + distillation, no bagging):"
echo "  python3 scripts/compare_arms.py $OUT/bag*_s42 $OUT/gold32_s42 \\"
echo "      --baseline models/ablate_full/r4distill_s42 \\"
echo "      --noise-from models/ablate_full/reg4_s42 models/ablate_full/reg4_s1337"

# 5. PER-PLANE SPECIALISTS — the constructive answer to the collapsed attention.
# The head cannot learn to look at the coronal plane for MCL (entropy 0.9994; MIL, temperature
# and pooling fixes all failed), so impose the specialisation instead of fighting for it.
# The three weakest findings are worth +0.031 of macro between them and ALL THREE are joint-line
# ligament/meniscus structures: LatMen 0.748 (coronal/sag), MCL 0.778 (CORONAL), ACL 0.806 (sag).
# Specialists are also cheaper (2 of 6 slots = a third of the encoder work) and far more
# DECORRELATED than two backbones on identical input — which is what select_ensemble.py needs.
# Judge them by ensemble contribution, not solo score: a coronal-only model SHOULD lose on
# axial findings and still earn its place.
run_arm cor_spec  42 --slots COR_FS,COR_T1
run_arm sag_spec  42 --slots SAG_FS,SAG_T1

echo
echo "then ask whether the specialists actually add anything:"
echo "  python3 scripts/select_ensemble.py models/ablate_full/r4distill_s42 \\"
echo "      models/ablate_full/dv3dist_s42 $OUT/cor_spec_s42 $OUT/sag_spec_s42 --max 6"

# 6. AUDIT ARMS (2026-08-25) — three augmentations were HARDCODED, always on, and NEVER measured.
# They have been shaping every result in this campaign without anyone choosing them:
#   * brightness/contrast jitter (p=0.5, alpha 0.9-1.1, beta +-25.5)
#   * translation +-4 px (p=0.5)
#   * anchor dropout (p=0.2, zero one of G)  <- now fires ON TOP of --anchor-bag
# The first is the suspect: on MRI, absolute intensity is DIAGNOSTIC — fluid is bright on
# fluid-sensitive sequences — and +-25.5 is 10% of the dynamic range, i.e. exactly the signal
# that separates effusion, synovitis and bone oedema.  A jitter that is standard on natural
# images may be actively destroying label information here.
# The third is now redundant: with bagging, a "bag of 6" is silently a bag of 5 one time in five.
run_arm nobc      42 --anchor-bag 6 --no-aug-bc       # is the intensity jitter costing us?
run_arm nogdrop   42 --anchor-bag 6 --no-aug-gdrop    # is anchor dropout double-counting bagging?
run_arm noshift   42 --anchor-bag 6 --no-aug-shift    # completes the set

# 7. BAGGING FOLLOW-UPS (2026-08-25), from the proposal round.
# bag6 ADOPTED at +0.0090 vs the champion on this layout, so the questions it opens are now
# the cheapest good ones we have:
run_arm bag6_s1337 1337 --anchor-bag 6   # SEEDS, REVISITED: they gave only +0.0011 before, but that
                                         # was deterministic training on FIXED inputs (rank corr
                                         # 0.953). Bagging makes training stochastic, so two seeds
                                         # now see different data and should decorrelate — which is
                                         # exactly what select_ensemble.py has been starved of.
run_arm_ep bag6_e16 42 16 --anchor-bag 6   # LONGER TRAINING, REVISITED: ep14 was NULL
                                         # (+0.0061) on a model shown the SAME 60 images every
                                         # epoch, where extra epochs only re-memorise. Bagging
                                         # injects fresh variety, which is the condition under
                                         # which longer schedules start paying.
