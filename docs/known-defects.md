# Known defects — RSNA Knee Abnormality Detection pipeline

> **HISTORICAL (pre-pivot).** This catalogue describes the 2026-08-20 tree that predates the
> SlotKnee-S pivot; the current pipeline is `src/slots.py`/`src/slotknee.py` + `scripts/`
> (see docs/slotknee_spec.md and docs/slotknee_runbook.md). Kept for the recovered-code record.

Ten defects that were each **reproduced and fixed** in the previous implementation of this pipeline.
The working tree was wiped on 2026-08-20 at 14:17, before anything was under version control. Backup
copies of the previous implementation were subsequently recovered into `recovered/` (see
[Where the fixed code survives](#where-the-fixed-code-survives)); they are **post-fix** snapshots, so for
most of these defects the *fix* survives — complete with the comment written when it was diagnosed —
while the broken code does not. Every claim below has been checked against that recovered source and
against the current `src/`.

The pipeline is being rebuilt from scratch right now. Every entry below carries a **Status in current
rebuild** field that was established by reading the current `src/` at the commit described in
[Verification basis](#verification-basis) — not by assumption. Three labels are used:

| Label | Meaning |
| --- | --- |
| **Already avoided** | The current code cannot exhibit this defect. |
| **Present** | The current code exhibits this defect, or an equivalent of it, today. |
| **Not yet implemented** | The rebuild has not reached this area; the defect is a trap waiting ahead. |

Read the status column first. Six of the ten defect classes are live in the rebuild right now.

## Summary

| # | Defect | Status in current rebuild |
| --- | --- | --- |
| 1 | [Plane mask wipeout — training on all-zero images](#1-plane-mask-wipeout) | **Not yet implemented**, but its *failure mode* is **present** by another route |
| 2 | [Non-contiguous tensors from transposed numpy views](#2-non-contiguous-tensors) | **Already avoided** — incidentally, with no guard |
| 3 | [Model selection on gold-subset macro-AUC](#3-model-selection-on-gold-subset-macro-auc) | **Present** (selection on AUC; made worse by a varying label denominator) |
| 4 | [Train/serve skew from a duplicated field list](#4-trainserve-skew) | **Already avoided** for the field list; a **different train/serve skew is present** |
| 5a | [AMP autocast enabled unconditionally](#5a-amp-autocast-enabled-unconditionally) | **Present** at three call sites |
| 5b | [`torch.compile` ungated on MPS](#5b-torchcompile-ungated) | **Not yet implemented** |
| 5c | [`nn.TransformerEncoder` nested-tensor fast path](#5c-transformerencoder-nested-tensor-fast-path) | **Not yet implemented** |
| 6 | [Horizontal flip mirrors knee anatomy](#6-horizontal-flip-mirrors-anatomy) | TTA **not yet implemented**; unconditional **train-time hflip is present** |
| 7 | [Runtime guard counted laptop sleep](#7-runtime-guard-counted-sleep) | **Not yet implemented** (no time budget exists) |
| 8 | [Label schema mismatches](#8-label-schema-mismatches) | **Present in part** — no normalisation, no column assertion, stale wrong label list still exported |
| 9 | [Synthetic data blended into real training](#9-synthetic-data-blended-into-real-training) | **Present** — dummy-label fallback in both `train.py` and `infer.py` |
| 10 | [Tensor cache poisoning](#10-tensor-cache-poisoning) | **Not yet implemented** (no cache exists) |

The unifying theme: **every one of these failed silently.** Not one produced a stack trace on the path
that mattered. Nine of ten produced a number that looked plausible. Design the rebuild so that
"looks plausible" is not the same as "is correct" — assert, don't hope.

---

## 1. Plane mask wipeout

**The worst defect in the set: the model trained on all-zero images for an entire run.**

### Symptom
Training ran to completion. Loss curves fell smoothly and looked entirely normal. Validation AUC hovered
near chance and would not improve regardless of backbone, learning rate, or augmentation. No exception,
no warning, no NaN.

### Root cause
The dataset called `normalize_series_df()`, which renames the series-metadata column
`Anatomical_Plane` → `Plane`. The very next block then read the **pre-normalisation** name:

```python
df = normalize_series_df(df)          # Anatomical_Plane -> Plane
...
plane = row.get("Anatomical_Plane", "")   # column no longer exists -> ""
if plane not in cfg.planes:               # "" matches nothing
    continue                              # every series masked out
```

`.get()` with a default made the missing column a silent empty string rather than a `KeyError`. Every
series failed the `cfg.planes` membership test, every series was masked out, and the sampler fell through
to its zero-fill path. The model received **all-zero image tensors** while the loss was computed against
the **real labels**.

### Why it was invisible
- A constant all-zero input is a perfectly learnable problem. The network simply learns the per-label
  base rate, and BCE loss descends toward the label-prior entropy in a smooth, convincing curve.
- Both the renaming call and the lookup were individually correct-looking; the defect lived only in
  their ordering.
- `dict.get(key, "")` and `Series.get(key, "")` are the standard defensive idiom, so the line read as
  careful code rather than as a bug.
- Zero-filling on empty input is also a standard defensive idiom. Two defensive idioms combined to
  destroy the dataset without a single complaint.

### How to prevent
1. **Never use `.get()` with a default for a column that must exist.** Index it directly so a rename
   raises `KeyError` at the first row.
2. **Assert non-empty selection, per study, at the point of selection:**
   ```python
   selected = df_series[df_series["Plane"].isin(cfg.planes)]
   assert len(selected) > 0, f"{study_id}: no series matched planes={cfg.planes}; " \
                             f"available={sorted(df_series['Plane'].unique())}"
   ```
   The assertion message must print what *was* available — that is what turns a five-hour hunt into a
   five-second read.
3. **Assert the batch is not blank.** One cheap guard in the training loop catches this entire class:
   ```python
   assert images.abs().sum() > 0, "all-zero image batch"
   assert images.std() > 1e-6, f"degenerate batch, std={images.std():.2e}"
   ```
4. **Do the renaming once, at load, and never keep both vocabularies alive.** Have exactly one function
   that reads `*_series.csv` and returns the canonical frame; nothing downstream should ever see the raw
   column names. See also defect 8 — this is the same disease.
5. **Look at the pixels.** A `save_debug_grid()` call that dumps the first batch of every run as a PNG
   would have made this a ten-second diagnosis.

**How it was actually fixed** (`recovered/`): the dataset reads `row["Plane"]` by direct indexing
(`recovered/local_20260813/src/dataset.py:63`) so a rename raises, and a `require_images` config
gate was added whose comment names this exact failure — *"Drop studies whose image directory is absent.
They would otherwise train as all-zero images paired with real labels"*
(`recovered/icloud/src/config.py:226-229`). That second guard is the one the current rebuild most needs;
see the status note below.

### Status in current rebuild
**Not yet implemented — but the failure mode is PRESENT by a different route.**

There is no plane handling in the rebuild at all: `Anatomical_Plane`, `Plane` and `normalize_series_df`
appear nowhere in `src/`, and `train_series.csv` is never read. `cfg.samples_per_plane` is declared in
`src/config.py:11` and passed by `scripts/train_ultimate_pipeline.sh:28,61`, but **no code reads it** —
`src/kaggle_data.py` walks every `.dcm` under the study directory and subsamples to `cfg.in_channels`
slices with `np.linspace`, ignoring plane entirely.

However, the *result* the defect produced — a zero image tensor paired with real labels — is reachable
in the current dataset by three separate paths, none of which raises:

- `src/kaggle_data.py:70-72` — missing study directory returns `torch.zeros(...)`, then falls through to
  `line 106-108` which returns the real labels alongside it.
- `src/kaggle_data.py:89-90` — zero slices collected produces a list of zero arrays.
- `src/kaggle_data.py:85` — a bare `except: continue` swallows **every** DICOM read failure. A systematic
  failure (wrong path layout, missing `pylibjpeg`/`gdcm` for compressed transfer syntaxes, permissions)
  silently yields zero slices for every study in the dataset.

Since `data_subset/` is currently empty, *every* study takes the `line 70-72` path today. Apply the
batch-level `images.abs().sum() > 0` assertion before this area of the code grows, and replace the bare
`except` with one that counts failures and raises when the failure rate crosses a threshold.

---

## 2. Non-contiguous tensors

### Symptom
Two different presentations from one cause. On CUDA: training was silently and substantially slower than
it should have been, with no error. On MPS: `batch_norm` backward crashed outright.

### Root cause
The augmentation stack transposed between `CHW` and `HWC` layouts. NumPy ufuncs default to `order="K"`,
which **preserves the input's memory layout** rather than producing a C-contiguous result. That
transposed layout therefore survived normalisation, `np.stack`, and `.astype()` untouched. The array
arriving at `torch.from_numpy` had an `NCHW` *shape* wearing `NHWC` *memory*:

```
shape   (N, 3, 320, 320)
strides (307200, 1, 960, 3)      # actual — channel stride 1: this is NHWC memory
strides (307200, 102400, 320, 1) # expected for true NCHW
```

`torch.from_numpy` faithfully wraps the strides, so the resulting tensor is non-contiguous.
`conv2d` accepts it and inserts a **silent full copy on every forward pass**; MPS `batch_norm` backward
does not accept it and dies.

Reproduced on the currently installed torch 2.8.0:

```
D2 shape (3, 320, 320) strides(elems) (1, 960, 3) C_CONTIGUOUS False
D2 torch is_contiguous: False
D2 after ascontiguousarray: True
```

### Why it was invisible
- Every shape assertion passes. `.shape` is `(N, 3, H, W)` exactly as intended — only `.stride()` reveals
  the problem, and nobody prints strides.
- On CUDA there is **no error at all**, only a throughput tax that reads as "this backbone is heavy".
- `order="K"` is NumPy's default and is not mentioned at the call site. It looks like ordinary arithmetic.
- The MPS crash surfaced in `batch_norm` backward, several layers away from the dataset code that
  actually caused it — the traceback pointed at the wrong file.

### How to prevent
1. **`np.ascontiguousarray()` at the dataset boundary**, as the last statement before tensor conversion.
   It is a no-op when the array is already contiguous, so it costs nothing in the good case.
2. **Assert contiguity in the collate function or the first line of `forward`:**
   ```python
   assert x.is_contiguous(), f"non-contiguous input: shape={tuple(x.shape)} stride={x.stride()}"
   ```
   Print the stride in the message — the stride *is* the diagnosis.
3. **Never trust a shape check to prove layout.** Any test covering the dataset must assert
   `.is_contiguous()`, not just `.shape`.
4. Be equally wary of `.T`, `.transpose()`, `.swapaxes()`, `.moveaxis()`, `einops.rearrange` and
   negative-stride slicing such as `img[::-1]` — every one of them returns a view, and every downstream
   ufunc will preserve it.

### Status in current rebuild
**Already avoided — incidentally, not by design. No guard is present.**

The current dataset never transposes in NumPy. It normalises each 2-D slice with Albumentations, converts
per slice with `ToTensorV2()` (`src/kaggle_data.py:51,57`), and joins with
`torch.cat(transformed_slices, dim=0)` at `src/kaggle_data.py:104`. `torch.cat` materialises a fresh
output tensor rather than returning a view, so the stride pathology cannot survive it. Verified on
torch 2.8.0 with the exact transform stack from `src/kaggle_data.py`:

```
per-slice shape torch.Size([1, 224, 224]) contig True
cat shape torch.Size([3, 224, 224]) contig True strides (50176, 224, 1)
```

This is fortunate rather than defended. The moment `torch.cat` is replaced with `np.stack` plus a
transpose — which is the natural refactor once plane/triplet sampling arrives, and is what the previous
implementation did — the defect returns with no test to catch it. **Add the `is_contiguous` assertion
now, while it passes**, so it exists before the refactor that breaks it.

---

## 3. Model selection on gold-subset macro-AUC

### Symptom
A real training run was killed by early stopping while **both training and validation loss were still
falling**. The trigger was a macro-AUC "non-improvement" of 0.0003 against a `min_delta` threshold.

### Root cause
Checkpoint selection and early stopping were both keyed on macro ROC-AUC computed over only the **5-6
gold-labelled studies** that landed in a given validation fold.

AUC is a rank statistic. Over a label with `p` positives and `n` negatives it can only take values on a
grid of spacing `1/(p·n)` — it is a step function, and it **cannot** move by less than one step. With
5-6 studies per fold that spacing is enormous, and because at that size only about half the labels have
both classes present, the macro average is taken over ~6 labels rather than 12, which doubles the
effective step again:

| Fold composition | Per-label step `1/(p·n)` | Macro step over 6 computable labels |
| --- | --- | --- |
| p=2, n=4 | 0.1250 | **0.0208** |
| p=3, n=3 | 0.1111 | **0.0185** |

That reproduces the previously recorded macro step of 0.017-0.021 exactly. The `min_delta` of 0.0003 is
roughly **55x smaller than the smallest change the metric is capable of expressing.**

The recovered source records the same conclusion from two different fold compositions, so treat the
figure as a range rather than a constant: `recovered/icloud/src/train.py:115-119` describes the mode
*"that stopped folds at epoch 8 because 0.7472 failed to beat 0.7475, a difference ~55x smaller than the
smallest rank swap the metric can represent"*, while `recovered/icloud/tests/test_train_selection.py:9-16`
records ~10 gold studies per fold giving steps of ~0.005-0.009 and a ratio of ~25x. **Between 25x and 55x
below the metric's resolution, depending on how many gold studies land in the fold.** The precise
multiplier does not matter; the sign of the comparison does. A threshold below
the resolution of its metric is not a threshold at all; it is noise amplification. The 0.0003 movement
was not a real regression — it could not have been, because no such movement exists on the grid. It was
floating-point residue from the varying set of computable labels.

Measured today from the shipped `train.csv` (see [Verification basis](#verification-basis)) — even using
**all 58** gold studies at once, the metric is still coarse:

| Label | pos | neg | step `1/(p·n)` |
| --- | --- | --- | --- |
| ACL | 24 | 34 | 0.0012 |
| MCL | 9 | 49 | 0.0023 |
| Medial Meniscus | 26 | 32 | 0.0012 |
| Lateral Meniscus | 23 | 35 | 0.0012 |
| Medial OA | 15 | 43 | 0.0016 |
| Lateral OA | 11 | 47 | 0.0019 |
| PF OA | 21 | 37 | 0.0013 |
| Effusion | 35 | 23 | 0.0012 |
| Synovitis | 27 | 31 | 0.0012 |
| Baker's | 12 | 46 | 0.0018 |
| Contusion | 19 | 39 | 0.0013 |
| Fracture | 18 | 40 | 0.0014 |

Best case — all 58 gold studies, all 12 labels computable — the macro-AUC step is ~1.2e-4. A `min_delta`
of 0.0003 is still only ~2.4 rank swaps. **There is no fold split of 58 studies for which a 0.0003
threshold is meaningful.**

### Why it was invisible
- 0.0003 *looks* like a sane epsilon. It is the kind of number that is chosen once and never questioned,
  and it is small enough to appear conservative.
- The AUC number itself is well-formed: it is in `[0, 1]`, it moves between epochs, it is not NaN.
  Nothing about it announces that its resolution is 0.02.
- Early stopping firing looks like the system working correctly. The run "completed"; it did not crash.
- The contradicting evidence — falling train *and* val loss — was in a different column of the same log
  and was never cross-checked against the stopping decision.

### How to prevent
1. **Select and early-stop on `val_loss`, computed over ALL ~260 validation studies**, not on AUC over
   the gold handful. Loss is continuous, dense, defined for every study, and moves monotonically with
   genuine improvement. Report AUC for information; do not let it drive control flow.
2. **If a rank metric must drive selection, derive `min_delta` from the metric's own resolution** rather
   than picking a constant:
   ```python
   steps = [1.0 / (p * n) for p, n in per_label_pos_neg if p and n]
   min_delta = (sum(steps) / len(steps)) / len(steps)   # one rank swap, macro-averaged
   ```
3. **Log the resolution alongside the metric** so it is impossible to miss:
   `val_auc=0.7431 (macro step 0.0208, 6/12 labels computable, n=6)`.
4. **Assert the threshold is meaningful at startup:**
   ```python
   assert cfg.min_delta >= macro_step, (
       f"min_delta={cfg.min_delta} is below the metric resolution {macro_step:.4f}; "
       f"early stopping would fire on numerical noise")
   ```
5. **Never let the macro denominator vary between epochs.** Fix the label set once — either require all
   12 to be computable or record which subset is in use — otherwise epoch-to-epoch AUC values are not
   comparable quantities and the deltas between them are meaningless.

**How it was actually fixed** (`recovered/`) — this is the most thoroughly engineered fix in the set and
is worth copying rather than reinventing:

- `selection_metric` became an explicit choice of `gold_auc` (legacy, kept only for reproducing old runs)
  / `gold_log_loss` / `val_loss` / `auto`, resolved by a **pure function**
  `resolve_selection_metric(...) -> EffectiveMetric` that takes floats and returns the metric plus the
  `min_delta` it must beat (`recovered/icloud/src/train.py:100-200`).
- `auto` **measures the fold's own AUC quantisation** via `metrics.auc_resolution()` and uses gold AUC
  only when a single rank swap moves it by no more than `cfg.gold_auc_resolution_target` (default 0.005),
  otherwise falling back. No study count is hard-coded, so the rule starts preferring gold AUC by itself
  once the gold panel is large enough.
- A companion `gold_studies_for_auc_resolution(target)` reports **how many gold studies would be needed**
  to reach the target resolution, and that number is printed in the epoch log
  (`recovered/icloud/src/train.py:745-752`).
- Selection has three outcomes, not two: *improvement*, *within resolution* (the metric cannot tell the
  two models apart, so no vote is cast), and *regression* (`recovered/icloud/src/train.py:229-237`).
- Because the policy is pure, it is tested with **synthetic metric sequences and no training run at all**
  (`recovered/icloud/tests/test_train_selection.py`, 649 lines). Structure the rebuild's selection logic
  the same way: a function of a handful of floats is testable; a decision buried in the epoch loop is not.

### Status in current rebuild
**PRESENT.**

- `src/train.py:104` — `if val_auc > best_auc:` selects the checkpoint purely on macro AUC.
- `src/train.py:48-54` — `validate()` computes macro AUC over the validation fold and wraps each label in
  `try/except ValueError: pass`. Labels with a single class present are **silently dropped from the
  average**, so the denominator (`len(aucs)` at line 54) changes between epochs. This is precisely the
  varying-denominator condition that produced the spurious 0.0003 delta. It is also the mechanism that
  turns a 12-label macro into a 6-label macro without any indication in the log.
- `cfg.early_stopping_patience` (`src/config.py:19`) is declared and is set explicitly by
  `scripts/train_ultimate_pipeline.sh:35,68,97` — but **no code reads it**. Early stopping is not yet
  implemented, so there is no `min_delta` in the rebuild yet. **This is the moment to add loss-based
  selection instead**, before patience logic is written against AUC.
- No `val_loss` is computed anywhere. `validate()` returns AUC only.

Related and worth fixing at the same time: with NaN labels (see defect 9 status) `val_auc` is `nan`, and
`nan > best_auc` evaluates to `False` on every epoch, so **no checkpoint is ever written and the run
still exits 0**. Verified: `float('nan') > 0.0` → `False`.

---

## 4. Train/serve skew

### Symptom
Every inference run crashed. After the crash was worked around, checkpoints trained under different
preprocessing were ensembled together without complaint.

### Root cause
`infer.py` kept its **own copy** of `PREPROCESSING_FIELDS` — the list of config fields that define the
preprocessing contract and that a checkpoint must agree on. The copy had drifted from the training-side
list in two directions at once:

- it listed `n_slices`, a field that **did not exist** on the config object, so every run raised on the
  attribute lookup;
- it **omitted** `samples_per_plane` and `triplet_gap`, two fields that genuinely change the pixels.

The omission is the dangerous half. Because those two fields were not part of the compared contract,
checkpoints trained with materially different preprocessing passed the compatibility check and were
averaged into a single ensemble.

### Why it was invisible
- The crash was loud and therefore *fixed the wrong thing*: adding `n_slices` to the config (or removing
  it from the list) made the error go away and made the module look healthy. The silent omission
  survived that fix untouched.
- A duplicated list stays correct for exactly as long as nobody edits either copy. It is correct at
  review time and wrong later, so code review does not catch it.
- Mismatched-preprocessing ensembles do not fail — they just score worse. The loss is invisible without
  a clean baseline to compare against.

### How to prevent
1. **Derive the contract from one source.** Define it once, next to the config, and import it in both
   `train.py` and `infer.py`. Never restate a list in two modules.
2. **Prefer derivation over enumeration.** Rather than hand-listing fields, mark them on the dataclass
   (`dataclasses.field(metadata={"preprocessing": True})`) and compute the list, so a new field joins the
   contract automatically instead of being forgotten.
3. **Assert every listed field exists, at import:**
   ```python
   missing = [f for f in PREPROCESSING_FIELDS if not hasattr(Config(), f)]
   assert not missing, f"PREPROCESSING_FIELDS names non-existent config fields: {missing}"
   ```
4. **Store the config *inside* the checkpoint** and compare on load, rather than trusting the caller to
   pass matching flags:
   ```python
   torch.save({"state_dict": model.state_dict(), "cfg": dataclasses.asdict(cfg),
               "labels": LABELS, "version": PIPELINE_VERSION}, path)
   ```
   On load, diff the stored preprocessing fields against the live config and **refuse to ensemble** on
   mismatch.
5. **Test it:** save a checkpoint under config A, attempt to load it under config B differing only in
   `samples_per_plane`, assert it raises.

**How it was actually fixed** (`recovered/local_20260813/src/infer.py:35-42`): a single
`PREPROCESSING_FIELDS` tuple —

```python
PREPROCESSING_FIELDS = (
    "planes", "samples_per_plane", "triplet_gap",
    "image_size", "preferred_slice_count", "backbone",
)
```

Note `backbone` is in the contract, because the normalisation mean/std is selected per backbone from
`model.NORMALIZATION_STATS` — *"ensembling checkpoints with different backbones without also matching
normalization would be the same class of silent skew as a resolution mismatch"*. The checkpoint stores
the config (`ckpt["cfg"]`, `ckpt["model_state"]`) and `load_model()` reconstructs the model **from the
checkpoint's own config** rather than the caller's (lines 45-52). Inference then builds a per-checkpoint
`fingerprint` from those fields and **refuses to ensemble** across differing fingerprints, printing the
differing values (lines 118-127).

### Status in current rebuild
**Already avoided for the duplicated field list — but a DIFFERENT train/serve skew is PRESENT.**

The good news: there is no duplicated field list. `src/infer.py:7-9` imports `Config`, `RSNA25DModel`,
`RSNADataset` and `KNEE_TARGETS` from the same modules `train.py` uses, and both build their `Config`
through the same `Config.from_args`. Single source. Preserve this property.

The bad news: **the preprocessing contract is not carried by the checkpoint and is not checked.**

- `src/train.py:106` saves a bare `model.state_dict()` — no config, no label list, no version.
- `src/infer.py:18,35` rebuilds `Config` from **CLI overrides only**; anything not passed silently takes
  the dataclass default.
- `scripts/train_ultimate_pipeline.sh:62` trains Phase 2 at `image_size=384`, but line 113 invokes
  inference with only `require_real_data` and `data_dir` set — so **inference runs at the default
  `image_size=224`** (`src/config.py:10`) against 384-trained weights. This is a live train/serve skew in
  the shipped pipeline script, and it will not raise: `load_state_dict` validates parameter shapes, and
  a CNN with global pooling accepts any input resolution.
- The same applies to `samples_per_plane`, `in_channels` and the normalisation constants — none is
  recorded, none is compared.

Two further silent-config hazards in the same area, both worth fixing while the checkpoint format is
still being designed:

- `src/config.py:38-39` — `Config.from_args` does `if not hasattr(cfg, k): continue`, **silently
  discarding any unrecognised key**. `scripts/train_ultimate_pipeline.sh:89` passes
  `folds_to_train="(${FOLD},)"`, which is not a `Config` field; it is dropped without a word, so Phase 4
  falls back to `--folds` default `[0]` and retrains fold 0 five times.
- Phase 4 writes to the same `models_dir` with the same `fold_{n}_best.pt` filename as Phase 2
  (`scripts/train_ultimate_pipeline.sh:67,96`), so those five runs each overwrite the previous
  checkpoint. Make `from_args` **raise** on an unknown key.

---

## 5. CUDA-only optimisations applied unconditionally

**This bug class recurred three separate times.** Each instance is a performance feature that is correct
on CUDA and fatal elsewhere, enabled by a condition that is not actually a device check. Treat the class,
not just the three instances: **every accelerator-specific optimisation needs one gate, in one place.**

### 5a. AMP autocast enabled unconditionally

#### Symptom
`conv2d` crashed on CPU. Devices other than CUDA silently ran in bf16.

#### Root cause
```python
with torch.autocast(device_type=..., enabled=(scaler is not None)):
```
`scaler is not None` is **always `True`** — a `GradScaler` object is constructed unconditionally and
merely carries an internal `enabled` flag; it is never `None`. The author's intent ("enable AMP when the
scaler is active") and the expression's meaning ("enable AMP when the scaler object exists") are not the
same condition. Verified on torch 2.8.0: `GradScaler('cpu', enabled=False) is None` → `False`, while
`.is_enabled()` → `False`. **The object's existence tells you nothing about whether it is on.**

#### Why it was invisible
- The expression reads as a deliberate, careful guard. It has the *shape* of a device check.
- The `GradScaler` itself was correctly gated, so the surrounding code looked right and grepping for
  `enabled=` found a line that appeared to handle the case.
- On CUDA — the platform the run matters on — the behaviour is correct, so it never showed up in the
  runs anyone cared about.

#### How to prevent
1. **Gate on the device, once:** `amp_enabled = (device.type == "cuda")`, computed in one place and
   passed everywhere. Never re-derive it from the presence of an object.
2. **Never infer state from object existence.** If a flag is wanted, pass a flag; `scaler.is_enabled()`
   is the honest expression if a scaler must be consulted.
3. **Assert dtype in a smoke test:** run one forward pass on CPU and assert the output is `float32`.
4. Watch for the sibling bug: `torch.amp.autocast(device_type)` **defaults to `enabled=True`**. Omitting
   the argument is the same defect written more briefly.

**How it was actually fixed** (`recovered/icloud/src/train.py:426-431`, and identically in
`recovered/local_20260813/src/train.py:68-73`) — the fix is a comment as much as a code change,
and the comment is the part worth carrying forward:

> `use_amp` gates autocast explicitly — it must NOT be inferred from `scaler is not None`, because a
> `GradScaler` instance is never `None` even when constructed with `enabled=False`.

The call sites became `with torch.autocast(device_type=device.type, enabled=use_amp):`, with `use_amp`
computed once and `torch.cuda.amp.GradScaler(enabled=use_amp)` fed from the same flag.

#### Status in current rebuild
**PRESENT — at three call sites.**

```
src/train.py:20   with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
src/train.py:38   with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
src/infer.py:49   with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
```

None passes `enabled=`, so autocast is **on for every device**. The ternary selects the autocast
*device_type* string; it does not disable anything. `src/train.py:96` gates the `GradScaler` correctly
(`enabled=device.type=='cuda'`) — which makes the omission on the autocast lines look intentional, the
same trap as before.

Verified on torch 2.8.0, CPU:

```
D5a autocast('cpu') is_enabled: True   conv out dtype: torch.bfloat16
D5a outside autocast dtype:            torch.float32
```

Two notes on how this manifests today, stated precisely because the previous behaviour was different:

- On this torch build, CPU `conv2d` under autocast **does not crash** — it silently computes in
  `bfloat16`. That is arguably worse than the original crash: local CPU runs now differ numerically from
  CUDA runs with nothing in the log to say so.
- On MPS (`torch.backends.mps.is_available()` → `True` on this machine) the ternary yields
  `device_type='cpu'` while the tensors live on `mps` — a mismatched context that is at best a no-op and
  at worst confusing to debug.

Fix: compute `amp_enabled = device.type == "cuda"` once, pass `enabled=amp_enabled` at all three sites,
and add a CPU smoke test asserting `float32` output.

### 5b. `torch.compile` ungated

#### Symptom
On MPS, every forward pass died. On CUDA, everything was fine.

#### Root cause
Inductor's Metal backend emitted a kernel requiring **49,152 bytes of threadgroup memory**, above
Apple's **32,768-byte** limit. The kernel cannot be dispatched, so the failure is total rather than
degraded. `infer.py` gated its `torch.compile` call behind a device check; `train.py` did **not**, at
**three separate call sites**.

#### Why it was invisible
- One module was fixed and the other was not. Having seen the guard in `infer.py`, a reader reasonably
  assumes the codebase handles it — a partial fix is worse than no fix, because it defeats the search.
- Three call sites means fixing the one in the traceback leaves two.
- The error surfaces as a Metal resource-limit message from deep inside generated code, with no obvious
  connection to the `torch.compile` line that produced it.

#### How to prevent
1. **One helper, one gate, zero direct calls:**
   ```python
   def maybe_compile(model, device):
       if device.type != "cuda":
           return model
       return torch.compile(model)
   ```
   Then `grep -rn "torch.compile" src/` must return exactly one hit — and that grep belongs in CI.
2. **Make the compile step fail-soft**, logging loudly, so a backend limit degrades to eager rather than
   killing an 8-hour run.
3. When a device-specific guard is added anywhere, **grep for every other call site of the same API in
   the same commit.** Partial fixes to this class are the norm, not the exception.

**How it was actually fixed** (`recovered/icloud/src/train.py:552-570`) — exactly the helper described
above, `maybe_compile(model, cfg, device)`: opt-in via `cfg.compile_model`, `device.type != "cuda"`
returns the model untouched, and the `torch.compile` call is wrapped in `try/except Exception` that warns
and falls back to eager. Its docstring records the diagnosis verbatim, including *"Training had no such
gate (three call sites), which made the default config unrunnable on Apple Silicon while inference was
fine. Same shape as the AMP-on-CPU bug."* — the previous author had already identified 5a and 5b as one
defect class.

#### Status in current rebuild
**Not yet implemented.** `grep -rn "compile" src/` returns nothing. When compilation is added for Kaggle
throughput, add it through a single `maybe_compile()` helper from the start — the recovered one can be
lifted almost unchanged.

### 5c. `TransformerEncoder` nested-tensor fast path

#### Symptom
On MPS, the model crashed inside the transformer whenever a padding mask was supplied.

#### Root cause
`nn.TransformerEncoder` silently switches to a nested-tensor fast path when it is given a
`src_key_padding_mask` and its inputs satisfy certain conditions. That path calls
`aten::_nested_tensor_from_mask_left_aligned`, which is **not implemented on the MPS backend**.

#### Why it was invisible
- The optimisation is opt-**out**, not opt-in. It is enabled by default and is not visible at the
  construction site — `nn.TransformerEncoder(layer, num_layers=4)` gives no hint that a second execution
  path exists.
- It only triggers when a padding mask is passed, so variable-length batches crash and fixed-length ones
  do not. That looks like a data problem, not a backend problem.

#### How to prevent
1. `nn.TransformerEncoder(..., enable_nested_tensor=False)` — always, in this project. The fast path's
   benefit is modest; its portability cost is total.
2. **Test with a padding mask.** A shape test that never passes a mask never exercises the failing path.
3. Cover both the ragged and the full-length case in the same test.

**How it was actually fixed** (`recovered/icloud/src/model.py:154-161` and `305-309`): every
`nn.TransformerEncoder` construction passes `enable_nested_tensor=False`, at both the per-plane slice
aggregator and the cross-plane fusion layer, with the reason stated inline at the construction site.

#### Status in current rebuild
**Not yet implemented.** `grep -rn "Transformer\|nested" src/` returns nothing. `src/model.py` is a CNN
backbone (`timm`) with GeM pooling and multi-sample dropout, with no sequence model over slices or
planes. If a per-slice transformer aggregator is added later — the natural next step for a 2.5-D
pipeline — pass `enable_nested_tensor=False` at construction.

---

## 6. Horizontal flip mirrors anatomy

### Symptom
The four laterality-specific labels — `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA` —
scored consistently worse than the eight non-lateral labels, and hflip TTA made them worse still rather
than better.

### Root cause
On **coronal** and **axial** knee views a horizontal flip exchanges the **medial** and **lateral** sides
of the joint. Four of the twelve labels name a side. Averaging a flipped forward pass into the prediction
without swapping those label pairs therefore averages each laterality label with the model's estimate of
its *opposite*, systematically pulling both toward their mean — precisely blurring the distinction those
labels exist to measure. The eight non-lateral labels (ACL, MCL, PF OA, Effusion, Synovitis, Baker's,
Contusion, Fracture) are unaffected, which is why the damage was localised.

### Why it was invisible
- Averaging more views is a near-universal win, so TTA is not where anyone looks for a regression.
- The macro-AUC average over 12 labels dilutes the damage: four labels degrading is largely offset by
  eight unchanged, so the headline number barely moves.
- Per-label AUC would have shown it immediately — but only if someone compared per-label AUC with TTA
  against per-label AUC without it.

### How to prevent
1. **Prefer shift TTA. It is translation, not reflection, so it preserves laterality and is safe.**
2. If hflip TTA is used at all, **swap the label pairs on the flipped pass before averaging**:
   ```python
   FLIP_PAIRS = [("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")]
   p_flip = predict(hflip(x))
   for a, b in FLIP_PAIRS:
       p_flip[:, [IDX[a], IDX[b]]] = p_flip[:, [IDX[b], IDX[a]]]
   pred = 0.5 * (predict(x) + p_flip)
   ```
3. **The same reasoning applies to train-time augmentation.** A random hflip without a corresponding
   label swap teaches the model that medial and lateral are interchangeable — it trains *away* the exact
   feature four labels depend on. Either swap the labels in the augmentation, or restrict hflip to planes
   where it is anatomically meaningless, or drop it.
4. **Always report per-label AUC, never only the macro.** Any change that helps 8 labels and hurts 4 is
   invisible in a single averaged number.
5. **Regression test:** assert that TTA does not reduce the AUC of any individual label by more than
   noise.

**How it was actually fixed** (`recovered/icloud/src/config.py:195-207`): `tta_hflip: bool = False` and
`tta_shift: bool = True`, with the reasoning recorded at the flag and the condition for ever turning it
on stated explicitly — *"Enable only together with a medial/lateral label swap on the flipped pass."*
The recovered config also notes that training applied **no** flip augmentation, which compounds the TTA
problem: a model never trained on mirrored images finds the flipped pass out-of-distribution as well as
anatomically wrong.

*Caution on the label count:* the recovered config contains two stacked comment blocks that disagree —
one says **5** laterality-specific labels and includes `PF OA`, the newer one says **4** and does not
(`recovered/icloud/src/config.py:190` vs `:196-197`). **Four is correct for this purpose.** PF OA is the
patellofemoral compartment, a third compartment rather than one side of a medial/lateral pair, so a
horizontal flip does not exchange it with anything. The swap list has exactly two pairs.

*Note (reasoning, not a previously reproduced finding):* on **sagittal** views a horizontal flip instead
mirrors the anterior-posterior axis, which is the axis that distinguishes ACL from PCL and locates the
patella. Flip is therefore questionable on all three planes for different anatomical reasons; only the
coronal/axial medial-lateral effect was reproduced and measured.

### Status in current rebuild
**TTA is not yet implemented — but unconditional train-time hflip is PRESENT.**

`src/infer.py` contains no TTA of any kind: a single forward pass per model, averaged across checkpoints
(`src/infer.py:44-52`). So the TTA form of this defect has not been reintroduced yet.

However, `src/kaggle_data.py:48` applies `A.HorizontalFlip(p=0.5)` to every training sample:

```python
A.Compose([
    A.Resize(self.cfg.image_size, self.cfg.image_size),
    A.HorizontalFlip(p=0.5),          # <-- mirrors medial <-> lateral on coronal/axial
    A.ShiftScaleRotate(...),
    ...
])
```

There is no label swap anywhere (`grep -rn "flip" src/` returns this one line), and because the dataset
does not track which plane a slice came from (defect 1 status), it **cannot** apply a plane-conditional
rule even if one were written. Half of all training samples for the coronal and axial planes are
currently presented with medial and lateral exchanged while the label vector is left untouched. This is
the same anatomical error as the TTA defect, applied at training time.

The plane metadata needed to fix this properly is available and unused: `train_series.csv` carries
`Anatomical_Plane` with values `Sagittal` (9,864), `Coronal` (8,609), `Axial` (5,898).

`A.ShiftScaleRotate` on the same line is fine — shift and small rotation preserve laterality.

---

## 7. Runtime guard counted sleep

### Symptom
A 9-hour multi-fold session terminated having completed only about 1.5 hours of actual compute, skipping
the remaining folds. It reported that the session budget was exhausted, and by its own measure it was.

### Root cause
The session budget was measured with `time.time()`, which tracks **wall-clock time including system
sleep**. The laptop slept overnight; the guard woke to find that its budget had elapsed and shut the run
down correctly according to a clock that had counted eight idle hours as work. `time.monotonic()` on
macOS excludes time spent asleep and is the correct clock for measuring elapsed compute.

### Why it was invisible
- The guard behaved exactly as designed. There is no bug in the control flow — only in the choice of
  clock, one token deep in an otherwise unremarkable line.
- The failure requires a sleep event to reproduce, so it never appears in short interactive runs. It only
  bites overnight, which is the run you most wanted to keep.
- `time.time()` is the reflexive default for anything time-shaped.

### How to prevent
1. **Use `time.monotonic()` for every elapsed-time measurement.** Reserve `time.time()` for timestamps
   that must be human-readable or comparable across processes.
2. **Log both clocks when the budget check fires**, so a discrepancy is self-diagnosing:
   `budget exhausted: monotonic=1.5h wall=9.2h` immediately names the problem.
3. **Checkpoint before the guard can fire**, so that budget exhaustion resumes rather than discards.
4. Add a lint rule or grep check: `time.time()` used in a subtraction is almost always wrong.
5. **Create exactly ONE guard per session and share it across folds.** This is a second, independent
   trap recorded in the same class, and the recovered source flags it in capitals: a guard constructed
   fresh inside each fold resets the clock every time, so an N-fold run can silently consume up to **Nx**
   the intended budget — defeating the entire purpose of the guard.

**The pre-fix code survives** — `recovered/local_20260813/src/utils.py:175-195` is the defective
`RuntimeGuard` itself, the only defect here whose broken form was recovered rather than its fix:

```python
def __init__(self, limit_hours: float) -> None:
    self.limit_seconds = limit_hours * 3600
    self.start = time.time()          # <-- counts system sleep

def elapsed(self) -> float:
    return time.time() - self.start
```

Its docstring already carried the one-guard-per-session warning, and the later
`recovered/icloud/src/train.py:1047,1054` shows the intended usage: a single `RuntimeGuard` built in
`main()` and threaded through `train_fold(..., guard=guard)`, with elapsed hours printed before each
fold. Changing both `time.time()` calls to `time.monotonic()` is the entire fix.

### Status in current rebuild
**Not yet implemented.** `grep -rn "time\.\|monotonic\|budget\|deadline" src/ scripts/` returns nothing —
there is no time budget, no runtime guard, and `time` is not imported. When a Kaggle session guard is
added (the 9- and 12-hour notebook limits make one necessary), use `time.monotonic()` from the first
line rather than converting later.

---

## 8. Label schema mismatches

### Symptom
A module would read `train.csv`, find **0 of 12** label columns, classify every row as non-gold, and
proceed. The report generator produced **no output at all and exited with status 0**.

### Root cause
The competition CSV ships human-readable label names with spaces and an apostrophe, while the code used
Python-identifier-safe names. Confirmed against the shipped file today:

```
$ head -1 ~/.cache/kagglehub/competitions/rsna-knee-abnormality-detection/train.csv
StudyInstanceUID,Report,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,
Effusion,Synovitis,Baker's,Contusion,Fracture
```

| In the CSV | In the code |
| --- | --- |
| `Medial Meniscus` | `Medial_Meniscus` |
| `Lateral Meniscus` | `Lateral_Meniscus` |
| `Medial OA` | `Medial_OA` |
| `Lateral OA` | `Lateral_OA` |
| `PF OA` | `PF_OA` |
| `Baker's` | `Bakers_Cyst` |

Any module that read `train.csv` **without** normalising found none of its expected columns. Because
gold-study detection was implemented as "does this row have all 12 label columns populated", zero matched
columns meant zero gold studies — a well-formed, entirely wrong answer. The report generator then had
nothing to report, wrote nothing, and exited cleanly.

### Why it was invisible
- The failure is a *count*, not an exception. "0 gold studies" is a valid number.
- Exit code 0 with no output reads as "nothing to do", which is indistinguishable from success in any
  script or CI that checks only the return code.
- The mismatch is subtle to the eye: `Medial OA` versus `Medial_OA` differ by one character.
- `Baker's` contains an apostrophe, which breaks naive quoting and encourages ad-hoc per-module renaming
  — which is how the vocabularies diverged in the first place.

### How to prevent
1. **One canonical label vocabulary and one normalisation function**, applied immediately at every CSV
   read. Nothing downstream of the loader may ever see a raw column name. (Defect 1 is the same disease
   in the series metadata: the fix is the same.)
2. **Assert the count immediately after loading — this single line prevents the whole failure:**
   ```python
   found = [c for c in LABELS if c in df.columns]
   assert len(found) == 12, f"expected 12 label columns, found {len(found)}: " \
                            f"missing={set(LABELS) - set(found)}; csv has {list(df.columns)}"
   ```
3. **Make "zero rows" an error, never a quiet success.** Any stage that produces no output must exit
   non-zero with a message naming what it looked for.
4. **Round-trip test:** load `train.csv`, normalise, denormalise, and assert the header matches
   `sample_submission.csv` exactly — including the apostrophe.
5. Note that the **submission header must use the CSV's own names.** `sample_submission.csv` ships
   `ACL,MCL,Medial Meniscus,...,Baker's,...`, so normalisation must be reversed on write.

### Status in current rebuild
**PRESENT in part.** The immediate lookup happens to work; every structural safeguard is missing.

What is correct: `KNEE_TARGETS` at `src/kaggle_data.py:30-34` uses the **CSV spelling**
(`"Medial Meniscus"`, `"PF OA"`, `"Baker's"`), which matches both `train.csv` and `sample_submission.csv`
byte-for-byte. `src/infer.py:58` writes the submission with these names, so the output header is right.

What is missing or wrong:

- **No normalisation function and no column assertion exist anywhere.** `src/kaggle_data.py:106-107` does
  `row[KNEE_TARGETS]` directly. The rebuild is currently one refactor to underscore names — the spelling
  used throughout the project brief — away from reintroducing the defect exactly.
- **A stale, wrong label list is still live in the module.** `src/kaggle_data.py:12-25` defines
  `TARGET_COLS` as **25 lumbar-spine columns** (`spinal_canal_stenosis_l1_l2`, …) from a different RSNA
  competition, followed at lines 27-28 by a comment acknowledging it is wrong. It is unused but exported,
  so `from src.kaggle_data import TARGET_COLS` succeeds and silently yields the wrong competition's
  labels. Two label vocabularies in one module namespace is the precondition for this entire defect
  class.
- **The gold-row filter does not exist.** `train.csv` holds **4,407 studies of which only 58 carry all 12
  labels**; the other 4,349 have every label empty. `src/train.py:70-78` reads the whole CSV and assigns
  folds with `df.index % 5` without filtering, so 98.7% of training rows carry `NaN` labels. Verified:
  `row[KNEE_TARGETS].values.astype(np.float32)` → all `nan`, and
  `BCEWithLogitsLoss()(zeros, nan_target)` → `nan`. Training loss is `nan` from the first batch, val AUC
  is `nan`, and — as noted under defect 3 — `nan > best_auc` is `False`, so the run completes all 8
  epochs, saves nothing, and exits 0. **This is defect 8's signature failure — a well-formed wrong answer
  with a clean exit — reproduced through the label *values* rather than the label *names*.**
- `df.index % 5` also does not group by study or stratify by label, so folds will not be reproducible or
  balanced once real labels are in place.

---

## 9. Synthetic data blended into real training

### Symptom
Training runs were contaminated with pure noise carrying confident labels. Metrics were unstable and
unreproducible between runs that appeared identically configured.

### Root cause
An optional external-dataset path was activated **purely because a directory existed** on disk — no
config flag, no explicit opt-in, no content validation. Its 25 volumes were random noise, but they
carried confident, expert-looking label vectors, and they were blended into real training at **4x
sample weight**. The presence of a directory is not consent.

### Why it was invisible
- The trigger is filesystem state, not configuration. The run's config was identical between a clean run
  and a contaminated one, so the two were indistinguishable in the log, in the command line, and in git.
- The labels looked like plausible expert annotations, so a spot check of the label file revealed nothing.
- 4x weighting meant 25 volumes acted like 100 — enough to move metrics substantially, but not enough to
  make training obviously diverge.

### How to prevent
1. **Never let filesystem state activate a code path.** Optional data sources require an explicit config
   flag; the directory check may only *validate* a flag that was already set.
   ```python
   if cfg.use_external_data:
       assert ext_dir.exists(), f"use_external_data=True but {ext_dir} is missing"
   ```
   Missing data with the flag set must **raise**, not silently skip. Present data with the flag unset
   must be **ignored**, not silently used.
2. **Validate content, not just existence.** Assert on the statistics of any external volume — an
   intensity histogram, a non-degenerate standard deviation, a plausible DICOM header. Pure noise is
   trivially detectable: `assert vol.std() > threshold and not is_uniform_random(vol)`.
3. **Log the provenance of every training sample count**, e.g.
   `train: 4407 real + 0 external (weight 1.0)` — printed every run, so contamination is visible in line
   one of the log.
4. **Never fabricate labels as a fallback.** A missing input file is an error. Fabricated data that
   reaches an optimiser is indistinguishable from real data once training starts.

**How it was actually fixed** (`recovered/icloud/src/train.py:597-606`) — the blend became opt-in behind
`cfg.use_mrnet` (default `False`), vetted at startup by `validate_mrnet_config()`, with the diagnosis
recorded at the call site: *"blending on mere directory existence silently mixed the synthetic
random-noise fixture into real training data, at external_weight — confident labels on pure noise."*
The external source was the public **MRNet** dataset (`recovered/icloud/src/config.py:236-247`), whose
`external_weight` of 4.0 was itself deliberate for *genuine* MRNet data — which is what made the
contamination so damaging when noise fixtures took the same path.

### Status in current rebuild
**PRESENT.**

There is no external-dataset path (`grep` finds no such logic), but the same defect class — fabricated
data entering a real run because a path was missing, gated by a bare `except` — is implemented in **both**
entry points:

```python
# src/train.py:69-75
try:
    df = pd.read_csv(os.path.join(cfg.data_dir, 'train.csv'))
except:
    print("WARNING: train.csv not found, generating dummy data for testing")
    df = pd.DataFrame({'StudyInstanceUID': [1, 2, 3, 4, 5]})
    for target in KNEE_TARGETS:
        df[target] = [0, 1, 0, 1, 0]
```

```python
# src/infer.py:23-27
except:
    print("WARNING: test.csv not found, generating dummy data for testing")
    test_df = pd.DataFrame({'StudyInstanceUID': [1, 2, 3]})
```

Points to note:

- The `except` is **bare**. It catches not only `FileNotFoundError` but `PermissionError`,
  `pd.errors.ParserError`, `KeyboardInterrupt` and `MemoryError` — so a transient or partial read
  failure on real data also lands in the fabricated-data branch.
- The fallback fabricates a **fixed alternating label pattern** `[0,1,0,1,0]` for all 12 labels, then
  trains for the full `cfg.epochs`, saves checkpoints to `cfg.models_dir`, and exits 0. Those checkpoints
  are indistinguishable on disk from real ones — and `src/infer.py:34-39` will happily ensemble them,
  since (defect 4) checkpoints carry no provenance.
- **`cfg.require_real_data` exists precisely to prevent this and is never read.** It is declared at
  `src/config.py:9` with default `True`, and `scripts/train_ultimate_pipeline.sh` sets
  `require_real_data=false` at lines 25, 58, 87 and 113 — but `grep -rn "require_real_data" src/` finds
  **only the declaration**. The guard is present in name, in the config, and in the shipped invocation,
  and absent from the code. This is the most misleading configuration in the repository: it reads as
  though the protection exists.
- `data_subset/` is currently **empty**, so this branch is what executes today. Any run started right now
  trains on five fabricated studies and reports success.

Fix: read `cfg.require_real_data` and raise when it is `True` and the CSV is absent; narrow the `except`
to `FileNotFoundError`; and move dummy-data generation into the test suite, where fabricated data
belongs, rather than into the training entry point.

---

## 10. Tensor cache poisoning

### Symptom
After a preprocessing bug was fixed, training continued to consume the **old, broken** tensors. **13,230
all-zero entries** were found on disk in the cache.

### Root cause
The cache key was derived only from the *preprocessing config fields* — image size, slice counts, and so
on. It did not incorporate anything about the *code* that turned DICOM into pixels. A fix that changed
the pixel content without changing any config value therefore produced an identical key, and every lookup
returned the stale entry. The bug outlived its own fix.

This is the compounding failure of defect 1: the plane-mask wipeout wrote thousands of all-zero tensors,
and the cache then faithfully served them long after the wipeout was corrected.

### Why it was invisible
- Cache hits are the desired behaviour. Fast epochs after a fix read as the fix working.
- Nothing distinguishes a stale entry from a fresh one at read time — both are well-formed tensors of the
  right shape and dtype.
- The fix was verified by re-reading the code, not by re-reading the pixels.

### How to prevent
1. **Refuse to store degenerate tensors.** The cheapest guard, and the one that would have prevented all
   13,230 entries:
   ```python
   if not np.any(arr):
       raise ValueError(f"refusing to cache all-zero tensor for {study_id}")
   ```
   Same check on read, so entries written before the guard existed are rejected rather than served.
2. **Include a code version in the cache key.** A `PREPROCESS_VERSION` constant that must be bumped by
   any change to the preprocessing path, or a hash of the preprocessing module's source. Keying on config
   alone is keying on half the inputs.
3. **Namespace the cache directory by that version**, so invalidation is a new directory rather than a
   deletion — old entries become unreachable without anything being destroyed.
4. **Provide a `--no-cache` flag** and use it whenever preprocessing changes.
5. **Log the hit/miss ratio each epoch.** A 100% hit rate immediately after a preprocessing change is the
   signal that something is wrong.

### Status in current rebuild
**Not yet implemented.** There is no cache: `grep -rn "cache" src/` returns nothing, and the only
persistence is `torch.save`/`torch.load` of model weights (`src/train.py:106`, `src/infer.py:36`).
`src/kaggle_data.py` re-reads and re-decodes every DICOM on every `__getitem__` call.

A cache will almost certainly be needed — `train.csv` holds 4,407 studies and `train_series.csv` holds
24,371 series, and decoding that on every epoch is not viable within a Kaggle session. Build the
all-zero refusal and the version-keyed path **into the first version of the cache**, not after the first
poisoning. Note the ordering hazard: the zero-tensor paths in `src/kaggle_data.py:70-72, 89-90` (defect 1
status) are live **now**, so a cache added on top of the current dataset would begin poisoning itself
immediately.

---

## Kaggle-specific notes

The target platform is **Kaggle Notebooks on Nvidia CUDA GPUs**. Three of these defects are
MPS/macOS-specific, which changes their priority **on Kaggle** but does not make them irrelevant —
they are all live on the local development machine, which is an Apple Silicon Mac with
`torch.backends.mps.is_available() == True` and no CUDA device.

**Lower priority on Kaggle, but still required for local development:**

- **5b (`torch.compile` on MPS)** — Inductor's 49,152-byte threadgroup requirement versus Apple's
  32,768-byte limit is a Metal-backend constraint. It cannot occur on CUDA. On Kaggle, `torch.compile` is
  a *desirable* optimisation. Still route it through a single `maybe_compile()` gate, because otherwise
  every local run must be edited before it will start — and edit-before-run is how ungated code gets
  committed.
- **5c (`TransformerEncoder` nested tensor)** — `aten::_nested_tensor_from_mask_left_aligned` is
  unimplemented on MPS; it works on CUDA. `enable_nested_tensor=False` costs almost nothing on CUDA and
  makes the model runnable in both places, so set it unconditionally rather than gating it.
- **7 (wall-clock versus monotonic)** — Kaggle sessions run on Linux containers that do not suspend, so
  the sleep discrepancy does not arise there. **But Kaggle imposes its own hard session limits (9h/12h),
  which makes a runtime guard mandatory rather than optional**, and that guard will be developed and
  debugged locally on a laptop that *does* sleep. `time.monotonic()` is correct on both platforms; there
  is no reason to write `time.time()`.

**Note that 5a is the opposite case — it is the CUDA-targeted variant that is dangerous *locally*.**
AMP is genuinely wanted on Kaggle. The defect is that it is not switched off anywhere else, so every
local CPU run silently computes in bfloat16 (verified above) and diverges numerically from the runs that
count. Gate on `device.type == "cuda"`.

**Higher priority on Kaggle than they were locally:**

- **1, 8, 9 (silent wrong data)** — a Kaggle notebook run is expensive and non-interactive. A run that
  trains on all-zero images, `NaN` labels, or fabricated studies burns a large fraction of a weekly GPU
  quota and reports success. The batch-level assertions are cheap insurance; add them before the first
  full run.
- **4 (train/serve skew)** — Kaggle inference notebooks are typically separate from training notebooks,
  with checkpoints passed as a dataset. That is a *wider* gap between train and serve than a single
  local repo, so the config-in-checkpoint discipline matters more, not less. The
  `image_size=384` train / `image_size=224` infer mismatch already present in
  `scripts/train_ultimate_pipeline.sh` is exactly the shape this takes.
- **10 (cache)** — 4,407 studies and 24,371 series cannot be re-decoded per epoch inside a session
  limit, so a cache is effectively mandatory. Kaggle's `/kaggle/working` persistence and dataset-mounting
  process also mean a poisoned cache can be **published as a dataset and reused across notebooks**,
  spreading the contamination beyond the run that created it.
- **3 (metric resolution)** — unchanged by platform. It is a property of having 58 gold studies, and it
  will be true on Kaggle exactly as it was locally.
- **6 (anatomical mirroring)** — unchanged by platform. It is a property of knee anatomy.

---

## Where the fixed code survives

Backup copies of the previous implementation were recovered into `recovered/` after the wipe. **This is
reference material, not a drop-in source** — it targets the old layout and the old data root, and it was
recovered from three different points in time, so the copies disagree with each other. Read it to see how
a defect was fixed; do not copy files wholesale into the rebuild.

| Copy | State | Contents |
| --- | --- | --- |
| `recovered/icloud/` | **Latest and most complete.** Post-fix. | `src/train.py` (1118), `src/labels.py` (974), `src/model.py` (528), `src/config.py` (368), `tests/test_train_selection.py` (649), plus `results/report/` |
| `recovered/local_20260813/` | Older. Post-fix for 4, 5a; **pre-fix for 7** | `src/` incl. `dataset.py`, `dicom_utils.py`, `infer.py`, `report.py`, `metrics.py`, `utils.py`, `pretrained_backbones.py`; `docs/`, `notebooks/`, `scripts/` |
| `recovered/gdrive/` | **Empty — every `.py` is 0 bytes.** | Filenames only. Still informative as a *module inventory*: `cache.py`, `schema.py`, `metrics.py`, `labels.py`, `dicom_utils.py`, `mrnet.py`, `report.py`, and eight test modules incl. `test_adversarial.py`, `test_infer_efficiency.py` |

Where to look for each fix:

| Defect | Fixed code |
| --- | --- |
| 1 plane mask | `local_20260813/src/dataset.py:63,110-122` (direct `row["Plane"]`, plane mask); `icloud/src/config.py:226-229` (`require_images`) |
| 3 selection | `icloud/src/train.py:100-200,229-237,745-752`; `icloud/tests/test_train_selection.py` |
| 4 train/serve skew | `local_20260813/src/infer.py:35-42,45-52,114-127` |
| 5a AMP | `icloud/src/train.py:426-431,485,694`; `local_20260813/src/train.py:68-73,87,167` |
| 5b `torch.compile` | `icloud/src/train.py:552-570` (`maybe_compile`) |
| 5c nested tensor | `icloud/src/model.py:154-161,305-309` |
| 6 hflip TTA | `icloud/src/config.py:195-207` (`tta_hflip=False`, `tta_shift=True`) |
| 7 runtime guard | **pre-fix** at `local_20260813/src/utils.py:175-195`; usage at `icloud/src/train.py:588,1047-1057` |
| 9 synthetic blend | `icloud/src/train.py:597-606`; `icloud/src/config.py:236-247` |
| 2, 8, 10 | **Not recovered.** `cache.py` and `schema.py` exist only as 0-byte filenames in `recovered/gdrive/`, and no recovered copy contains the dataset-boundary `ascontiguousarray` or the label-schema normaliser. For these three, this document is the only record. |

Two things the recovered source proves beyond the individual fixes:

1. **The previous author had already generalised defect 5 into a class.** `maybe_compile`'s docstring ends
   *"Same shape as the AMP-on-CPU bug."* The rebuild should carry a single device-capability gate rather
   than three independent ones.
2. **The fixes were written as prose at the point of decision.** Nearly every one is a short comment on
   the config field or call site explaining what went wrong and what condition would justify changing it
   back. That is why they were recoverable as knowledge and not just as code — and it is the cheapest
   defect-prevention practice in this whole document.

---

## Verification basis

Every "Status in current rebuild" claim above was checked on **2026-08-20** against the working tree at
commit `b889ea7` ("Expand model with GeM pooling and Multi-Sample Dropout"), branch `main`, clean.
`src/` at that point is 391 lines across six files:

```
src/__init__.py       1
src/config.py        47
src/model.py         59
src/infer.py         64
src/train.py        110
src/kaggle_data.py  110
```

`src/` is being actively rewritten, so **line numbers cited here will drift.** Prefer the
quoted code over the line reference when the two disagree.

Empirical checks were run on the installed **torch 2.8.0** (CPU/MPS; `torch.cuda.is_available()` is
`False` on this machine) and against the read-only competition CSVs in
`~/.cache/kagglehub/competitions/rsna-knee-abnormality-detection/`:

| Claim | Method | Result |
| --- | --- | --- |
| CSV label spelling (defect 8) | `head -1 train.csv`, `head -1 sample_submission.csv` | `Medial Meniscus`, `PF OA`, `Baker's` — spaces and apostrophe confirmed in both |
| Gold-study count (defects 3, 8) | count rows where all 12 label columns are non-null | 58 of 4,407 studies; 4,349 have no labels at all |
| AUC step size (defect 3) | `1/(p·n)` per label over the 58 gold studies | 0.0012–0.0023 per label; macro ≈ 1.2e-4 |
| AUC step at fold scale (defect 3) | `1/(p·n)` for p=2,n=4 and p=3,n=3, macro over 6 computable labels | 0.0208 and 0.0185 — reproduces the recorded 0.017–0.021 |
| `NaN` labels reach the loss (defect 8/9) | `astype(np.float32)` on an unlabelled row, then `BCEWithLogitsLoss` | all-`nan` target, `loss = nan`; `nan > 0.0` is `False` |
| Transposed-view strides (defect 2) | `hwc.transpose(2,0,1)` then a ufunc | strides `(1, 960, 3)`, `C_CONTIGUOUS False`, torch `is_contiguous False`; `np.ascontiguousarray` restores it |
| Current dataset is contiguous (defect 2) | the exact `A.Compose` + `ToTensorV2` + `torch.cat` path from `src/kaggle_data.py` | `contig True`, strides `(50176, 224, 1)` |
| Autocast default (defect 5a) | `with torch.amp.autocast('cpu')` around a `Conv2d` | `is_autocast_enabled` `True`, output dtype `torch.bfloat16` (vs `float32` outside) |
| `GradScaler` is never `None` (defect 5a) | `torch.amp.GradScaler('cpu', enabled=False)` | object is not `None`; `.is_enabled()` is `False` |
| Plane values available (defects 1, 6) | value counts of `Anatomical_Plane` in `train_series.csv` | `Sagittal` 9,864 / `Coronal` 8,609 / `Axial` 5,898, over 24,371 series |

Absence claims (`torch.compile`, `TransformerEncoder`, TTA, time budget, cache, plane handling) were each
established by `grep -rn` over `src/` and `scripts/` returning no matches.

The previous implementation's own account of these defects was then read from the recovered backups in
`recovered/` (see the previous section), which appeared in the working tree during this review. Those
copies corroborate defects 1, 3, 4, 5a, 5b, 5c, 6, 7 and 9 directly — in eight cases through the
surviving fix and its explanatory comment, and in the case of defect 7 through the surviving *defective*
code. Defects 2, 8 and 10 are recorded here only.

`recovered/` and `scripts/prepare_data_subset.py` appeared mid-session and are not the work of this
review; nothing outside `docs/` was created or modified in producing this document.

## SSL kernel pretrains on the pre-coverage cache (flagged 2026-08-29, RESOLVED 2026-09-02)

`slotknee-ssl` attaches only `felipedeleon11/slotknee-cache-builder`, so its CPU
pre-flight resolves the cache to `.../slotknee-cache-builder/slots_P224` — the old
224/3x3 v2 cache, not `slots_P224_g10t1`, the coverage layout every adopted model is
actually trained on. The kernel finds its cache with a first-match `walk_find` for
`train_index.json`, so attaching both caches would make the choice order-dependent;
picking one means *replacing* the source, not adding to it.

This is **not** a bug. MAE is label-free and the old cache holds real knee MRI slices;
the docstring shows it was a deliberate choice made before the coverage layout was
adopted. But the encoder would be pretrained on a different slice distribution than it
is fine-tuned on, which is the same family of drift as the reg4 backbone bug (that one
*was* broken — weights could not load — and is fixed).

**Decision deferred to Felipe**: swapping `kernel_sources` to
`felipedeleon11/slotknee-cache-g10t1` changes an unmeasured variable in an experiment
that has not been run. SSL is priority 2 behind the ablation arms. Do not change it
silently as part of unrelated work.

**Resolved 2026-09-02** (entry corrected 2026-09-08): `kaggle/ssl_kernel/kernel-metadata.json`
now attaches only `felipedeleon11/slotknee-cache-g10t1` — the coverage layout every adopted model
is fine-tuned on (see the kernel docstring). The S5 data-mix requirement (all planes/sequences of
the coverage cache, no natural-image replay) is met. The kernel has still never produced an
encoder: `kaggle/.logs/slotknee-ssl/slotknee-ssl.log` is the v1 P100 fast-fail and v5 is
"not launched" (runbook:105).


## train-final loaded plain-DINOv2 weights into the reg4 backbone (FIXED 2026-09-07)

`kaggle/train_kernel_final/slotknee_train_final.py` hardcoded
`weights/vit_small_patch14_dinov2.lvd142m.safetensors` while its RECIPE passes
`--backbone vit_small_patch14_reg4_dinov2.lvd142m`. The register variant's `pos_embed` has a
different token count, so timm's `resample_abs_pos_embed` raised
`shape '[1, 37, 37, -1]' is invalid for input of size 526080` on the kernel's first-ever GPU run.
Fixed in v11: the weights file is derived from the recipe's `--backbone`. Lesson recorded in the
runbook: a CPU pre-flight validates mounts, not the model build — the encoder-load path only
runs on a GPU.

## Submission entries cannot be selected by environment variable from the UI (design, 2026-09-07)

Kaggle's Save & Run All has no environment panel, so `SK_CKPT_GLOB` / `SK_TTA` are reachable only
from API pushes, which land on a banned P100. Resolution: one kernel per entry with the defaults
baked in — `slotknee-submit` (accuracy) and `slotknee-submit-eff` (efficiency). Any future
per-entry switch must follow the same pattern.
