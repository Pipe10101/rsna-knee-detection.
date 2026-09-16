#!/usr/bin/env bash
# End-to-end smoke test for SlotKnee-S pipeline
# Should run in < 15 mins and < 6 GB RSS on local machine

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA_DIR="data_subset"
CACHE_DIR="cache/slots_smoke"   # NEVER cache/slots_P224 (the shared full cache)
OUT_DIR="models/slotknee_smoke"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "=== 1. Building Cache (Limit 48) ==="
if [ -f "$CACHE_DIR/train_index.json" ]; then
    echo "Cache $CACHE_DIR already present; skipping build."
else
    python3 scripts/build_slot_cache.py --data-dir "$DATA_DIR" --split train --out "$CACHE_DIR" --limit 48 --P 224
fi

echo "=== 2. Training (Fold 0, 2 Epochs, BS 4) ==="
# We assume no LLM CSV is present to test the regex_fallback labels
python3 scripts/train_slotknee.py \
    --cache "$CACHE_DIR" \
    --data-dir "$DATA_DIR" \
    --folds 0 \
    --epochs 2 \
    --bs 4 \
    --lr-head 3e-4 \
    --lr-backbone 5e-5 \
    --trainable-blocks 4 \
    --out "$OUT_DIR" \
    --seed 42 \
    --amp auto \
    --token-pooling \
    --max-studies 48

echo "=== 3. Inference ==="
python3 scripts/infer_slotknee.py \
    --data-dir "$DATA_DIR" \
    --ckpt "$OUT_DIR/fold_0_best.pt" \
    --out submission.csv \
    --P 224 \
    --bs 4

echo "=== 4. Validation ==="
python3 -c "
import pandas as pd
df = pd.read_csv('submission.csv')
sample = pd.read_csv('$DATA_DIR/sample_submission.csv')
assert list(df.columns) == list(sample.columns), 'Columns mismatch'
assert list(df.iloc[:, 0]) == list(sample.iloc[:, 0]), 'Row set/order mismatch'
vals = df.iloc[:, 1:].values
assert (vals > 0.0).all() and (vals < 1.0).all(), 'Values not in (0, 1)'
print('submission.csv matches sample_submission.csv rows/columns; values in (0, 1).')
"

echo "Smoke test completed successfully."
