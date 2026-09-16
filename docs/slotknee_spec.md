# SlotKnee-S — build spec (module contract)

Read `docs/research_pivot_efficient_model.md` for the *why*. This file is the *what*: exact
file ownership, interfaces, array layouts and constraints. Every module is built against this
and nothing else. If an interface has to change, change it HERE first and record it in the
module's build notes (§8).

## 0. The objective (one number)

The efficiency prize scores `Eff = AUC/(0.5 − maxAUC) + seconds/32400`, minimised, with
maxAUC ≈ 0.952. Equivalently we **maximise**

    J = macroAUC − inference_seconds / 71_700        (0.01 AUC == 717 s of runtime)

for a ~1,300-study test set on a Kaggle T4 (16 GB, fp16, 2 vCPU, 9 h cap). Every design
decision is judged on J. "Most efficient model possible" means: the smallest compute and
memory that does not lose measurable AUC (> 2× the measured noise floor).

## 1. Hard constraints

- **Runs on the developer's laptop**: Apple M5 Pro, 24 GB RAM, MPS backend, ~18 GiB free disk (2026-08-24; shrinking — check `df -h` before caching),
  Python 3.9, torch 2.8, timm 1.0.28, pydicom, cv2, albumentations, sklearn, pandas. No CUDA.
  `bf16` is NOT available on Kaggle T4 (sm_75) — resolve autocast dtype from the device.
- **Offline**: DINOv2-S weights are in the HF cache as
  `~/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m`; on Kaggle they
  are attached as a dataset. Never call the network at runtime. Use
  `src/model._resolve_weights_file` / timm `pretrained_cfg_overlay={'file': ...}` when a
  path is given; fall back to timm's normal cache lookup otherwise.
- **Low memory**: no process may hold more than ~4 GB RSS during cache building or
  inference; training ≤ 12 GB. The cache is memory-mapped; never `np.load` a whole shard.
- **Local data**: `data_subset/` has 649 train studies on disk (`train_images/<study>/<series>/*.dcm`),
  `train.csv` (4,407 rows, `Report`), `train_gold.csv` (58 labelled), `train_series.csv`
  (`StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane`),
  `test_images/` (3 studies), `test_series.csv`, `sample_submission.csv`. All local DICOMs are
  uncompressed; the real Kaggle data also has JPEG-Lossless / JPEG2000 (pylibjpeg + openjpeg
  are installed). Filenames are SOP UIDs — NOT in anatomical order.
- **No git**: do not commit, push, stash, or create branches/worktrees. Do not deploy to
  Kaggle. Edit files only.
- **Ownership**: each module is confined to the files listed for it in §2. Do not edit `src/labels.py`,
  `src/train.py`, `src/model.py`, `src/kaggle_data.py`, `src/infer.py`, `src/soup.py`,
  `src/config.py`, or existing tests. Import from them freely.
- **Tests**: each module ships a `tests/test_<module>.py` that runs in < 60 s on CPU
  with the local subset (use ≤ 8 studies) and passes with `python3 -m pytest tests/test_<module>.py -q`.
  Do not break the existing suite (`python3 -m pytest -q -x`; 455 tests green on 2026-08-24).

## 2. Files by module

| Module | Files (create/edit) | Must not touch |
|---|---|---|
| A — data & cache | `src/slots.py`, `scripts/build_slot_cache.py`, `tests/test_slots.py` | everything else |
| B — model | `src/slotknee.py`, `tests/test_slotknee.py` | everything else |
| C — labels & folds | `src/llm_labels.py`, `src/folds.py`, `tests/test_llm_labels.py`, `tests/test_folds.py` | everything else |
| D — train & infer | `scripts/train_slotknee.py`, `scripts/infer_slotknee.py`, `scripts/smoke_slotknee.sh`, `tests/test_slotknee_pipeline.py` | everything else |
| R — review & efficiency | `docs/slotknee_review.md`, `scripts/bench_slotknee.py` | any `src/` file |

## 3. Data contract (module A)

### 3.1 Sequence slots

Six slots, index order fixed:

    0 SAG_FS   Sagittal, Fluid_Sensitive=1
    1 COR_FS   Coronal,  Fluid_Sensitive=1
    2 AX_FS    Axial,    Fluid_Sensitive=1
    3 SAG_T1   Sagittal, Fluid_Sensitive=0
    4 COR_T1   Coronal,  Fluid_Sensitive=0
    5 AX_T1    Axial,    Fluid_Sensitive=0

Source of plane/contrast: the series CSV (`train_series.csv` / `test_series.csv`; at test
time Kaggle replaces `test_series.csv` with the real descriptors). If a series is missing
from the CSV, derive plane with `kaggle_data.derive_plane(ImageOrientationPatient)` and
contrast from `ScanOptions`/`SeriesDescription` (fat-sat / STIR / T2 / PD → fluid-sensitive)
with TR/TE fallback (TR>800 & TE>60 → T2; TR>800 & TE≤60 → PD; else T1). If several series
match a slot, take the one with the most slices (tie → smallest UID). Missing slot → zeros
and `mask[s]=0`.

### 3.2 Slice selection within a slot

`paths, plane = kaggle_data.order_series(files)` (geometry-sorted). With `n` slices,
trim `trim_frac=0.15` from each end, place `G=3` anchors with `np.linspace` over the trimmed
range (G=1 → centre), and take `T=3` physically adjacent slices `[a-1, a, a+1]` (clamped)
per anchor. Configurable G, T, trim_frac. Only these files are pixel-decoded.

### 3.3 Crop, resample, intensity

Per slice: read `PixelSpacing` (row, col mm) and `Rows/Columns` from the header; crop a
centred window of `crop_mm=140` (px = round(mm / spacing) per axis); if the window exceeds
the image, use the whole image and count it in the builder's report. Resize to `P×P`
(`P=224` default; P must be a multiple of 14) with `cv2.INTER_AREA`. Intensity: per-slot
1st–99th percentile over the selected slices → scale to 0..255 `uint8`.

### 3.4 Laterality

If the knee is RIGHT: flip columns of COR/AX slots; reverse anchor order in SAG slots.
Side = DICOM `Laterality` if present, else the sign of the patient-x coordinate of the
image centre (`IPP + r·Δc·Nc/2 + d·Δr·Nr/2`; x<0 → right), median over the study's series;
|x| < 20 mm → unresolved → no flip. Flag `laterality=True`.

### 3.5 Public API (`src/slots.py`)

```python
SLOT_NAMES: list[str]                      # 6 names above
def index_study(study_dir, series_df=None) -> StudyIndex
    # header-only pass (stop_before_pixels). StudyIndex has: .slots: dict[slot_idx -> list[path]]
    # (ordered paths of the chosen series), .side ('L'|'R'|None), .spacing per series, .n_files
def select_slices(index, G=3, T=3, trim_frac=0.15) -> dict[slot_idx -> list[list[path]]]   # G groups of T paths
def build_study_tensor(study_dir, series_df=None, P=224, crop_mm=140.0, G=3, T=3,
                       laterality=True) -> tuple[np.ndarray, np.ndarray, dict]
    # returns (x uint8 [6, G, T, P, P], mask uint8 [6], info dict with side, n_decoded, n_crop_fallback, ms)
```

`T` is the channel axis fed to the encoder (T=3 → RGB-like triplet).

### 3.6 Cache (`scripts/build_slot_cache.py`)

    python3 scripts/build_slot_cache.py --data-dir data_subset --split train --out cache/slots_P224 \
        [--limit N] [--P 224] [--crop-mm 140] [--G 3] [--T 3] [--workers 4]

Writes `cache/slots_P224/train_x.u8` — a flat `np.memmap` of shape `[N, 6, G, T, P, P]`
uint8 (opened with `mode='r+'`, written one study at a time; never materialised in RAM),
`train_mask.u8` `[N, 6]`, `train_index.json` (`{"studies": [uid...], "P":224, "G":3, "T":3,
"crop_mm":140, "version":"slots-v1", "stats": {...}}`). Resumable: skip studies already
marked done in the index. Multiprocessing with `workers` processes, each bounded to one
study in flight. Per-study budget ≈ 27 decoded 640² int16 slices ≈ 22 MB. Report at the end:
studies/s, ms/study, crop fallbacks, missing-slot histogram, peak RSS (`resource.getrusage`).
Also `--split test` for `test_images/` + `test_series.csv`.

Reader helper (in `src/slots.py`):

```python
class SlotCache:
    def __init__(self, dir, split="train"): ...   # memmaps; .uids, .P, .G, .T, .N
    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray]   # (x [6,G,T,P,P] uint8 view, mask [6])
    def row_of(self, uid) -> int
```

## 4. Model contract (module B, `src/slotknee.py`)

```python
class SlotKneeS(nn.Module):
    def __init__(self, P=224, T=3, d=256, n_labels=12, n_slots=6,
                 # library default stays plain DINOv2-S so in-flight ablation arms remain
                 # comparable; the ADOPTED production encoder is the register variant
                 # vit_small_patch14_reg4_dinov2.lvd142m (+0.0220 pooled OOF, 2026-08-24),
                 # which the production kernels pass explicitly.
                 backbone="vit_small_patch14_dinov2.lvd142m", pretrained_path=None,
                 trainable_blocks=4, pool="cls_mean", encoder_chunk=64,
                 grad_checkpointing=False, drop_path=0.05, head_dropout=0.1)
    def forward(self, x, mask) -> torch.Tensor
        # x: uint8 or float tensor [B, 6, G, T, P, P]; mask: [B, 6] (1 = slot present)
        # returns logits [B, n_labels]
    def slot_attention(self) -> torch.Tensor   # last forward's [B, n_labels, 6] attention (for inspection)
    def param_groups(self, lr_backbone, lr_head, llrd=0.8) -> list[dict]   # layer-wise LR decay
```

- Normalisation lives INSIDE the model as buffers read from the timm pretrained cfg
  (`mean/std`), so uint8 input is `x.float()/255` then normalised. T≠3 adapts the patch
  embedding via timm `in_chans` (sum/repeat), never by dropping weights.
- Encoder: timm `vit_small_patch14_dinov2.lvd142m` with `img_size=P`, `num_classes=0`,
  `global_pool=''`. Pool: concat(CLS, mean of patch tokens) → Linear(768→d) + LayerNorm + GELU.
  Freeze all blocks except the last `trainable_blocks` (and final norm). Frozen blocks run
  under `torch.no_grad()` to save activation memory.
- All `B·6·G` images go through the encoder in chunks of `encoder_chunk` images (memory
  bound), then reshaped to `[B, 6, G, d]`, mean over G → `[B, 6, d]`, add learned slot
  embedding `[6, d]`.
- Head: learned queries `[n_labels, d]`; attention logits `q·h/√d` masked with `mask`
  (−inf where absent; if a study has NO slots, fall back to uniform); context `[B, n_labels, d]`;
  per-label linear (`d→1`, separate weights per label, implemented as einsum) → logits.
- `param_groups` returns backbone blocks with LLRD (deepest trainable block at `lr_backbone`,
  each earlier ×llrd), head at `lr_head`.
- Must build and run a forward of `B=2, G=3, T=3, P=224` on CPU in < 10 s in the test,
  output shape `[2,12]`, masked slot receives exactly zero attention, gradient flows only to
  the last `trainable_blocks` blocks + head, and a `P=252` build works (18×18 patches).

## 5. Labels & folds contract (module C)

`src/llm_labels.py`

```python
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
          "Effusion","Synovitis","Baker's","Contusion","Fracture"]
def load_llm_labels(paths: list[str], blend="mean") -> pd.DataFrame   # StudyInstanceUID + 12 prob columns in [0,1]
    # accepts any CSV whose columns match LABELS (case/space-insensitive); 0.5 means "not addressed"
def cell_weights(p: np.ndarray) -> np.ndarray         # 2*|p-0.5|, so 0.5 -> 0 weight
def fill_silent_synovitis(df) -> pd.DataFrame           # where Synovitis==0.5 and Effusion!=0.5: Synovitis = 0.5 + 0.5*(Effusion-0.5)
def build_targets(llm_df, gold_df=None, gold_weight=8.0, min_weight=0.0) -> pd.DataFrame
    # one row per study: StudyInstanceUID, y_<label> (float target), w_<label> (weight), is_gold (bool)
    # gold rows override LLM values with hard 0/1 and weight gold_weight on every cell
```

The public CC0 label files are Kaggle datasets (not on this machine):
`stevenleehans/rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`), Pilkwang Kim's
`rsna-knee-llm-labels`, `lixin73` GPT labels. Code must work when NONE of them is present:
fallback = `src.labels.label_dataframe(...)`-style regex targets via
`train.build_derived_from_reports`-compatible columns — implement
`regex_fallback(train_csv, data_dir) -> DataFrame` by calling `src.labels` (do not write a
new parser). Tests use a small synthetic CSV.

`src/folds.py`

```python
def scanner_fingerprint(study_dir) -> str   # Manufacturer|ManufacturerModelName|SoftwareVersions|MagneticFieldStrength|ImagingFrequency (whatever exists after the 86-tag allowlist; header-only read of ONE file)
def report_group(report_text) -> str        # sha1 of normalised text (lowercase, whitespace collapsed)
def assign_grouped_folds(df, n_folds=5, seed=42, group_col="group", strat_col=None) -> np.ndarray
    # StratifiedGroupKFold on group_col; strat on number of positives (y>0.5 count) bucketed 0/1-2/3-4/5+ unless strat_col given
def build_groups(df, data_dir, image_dir="train_images") -> pd.Series   # fingerprint + report hash -> group id
```

Gold rows keep their existing `fold` column from `train_gold.csv` when present.

## 6. Train & infer contract (module D)

`scripts/train_slotknee.py` (args: `--cache cache/slots_P224 --labels <csv...> --data-dir
data_subset --folds 0 --epochs 6 --bs 8 --lr-head 3e-4 --lr-backbone 5e-5 --trainable-blocks 4
--out models/slotknee --seed 42 --amp auto --max-studies N --limit-minutes M`):

- Dataset over `SlotCache` (zero-copy memmap rows); targets/weights from
  `llm_labels.build_targets`; folds from `folds.assign_grouped_folds` (gold keeps its fold).
- Augmentation on uint8 tensors, cheap and label-safe: random group dropout (drop one of G
  with p=0.2), per-slot brightness/contrast ±10 %, small translation ±4 px; NO hflip.
- Loss: `train.MaskedBCEWithLogitsLoss` with soft targets and per-cell weights.
- AdamW, cosine with 1-epoch warmup, `model.param_groups(...)`, grad-clip 1.0,
  autocast: cuda→fp16 with GradScaler, mps→fp32 (or fp16 if it verifies finite on a probe
  batch), cpu→fp32. EMA of weights (decay 0.998) evaluated each epoch.
- Validation every epoch on the fold's held-out rows: masked BCE + macro AUC on derived
  targets (cells with w>0) AND separately on gold rows; log both; select on val AUC over
  derived cells (800+ rows) — gold AUC is reported only.
- Saves `fold_{k}_best.pt` (state_dict + config json), `oof_fold_{k}.npz` (uids, logits,
  targets, weights), `fold_{k}_log.json`. Resumable per fold. Respect `--limit-minutes`.
- Peak RSS and MPS/CUDA peak memory printed per epoch.

`scripts/infer_slotknee.py` (args: `--data-dir data_subset --ckpt models/slotknee/fold_0_best.pt
[more ckpts] --out submission.csv --P 224 --bs 4`):

1. Copy `sample_submission.csv` → `submission.csv` FIRST.
2. For each test study: `slots.build_study_tensor` on the fly (header-first; no cache
   written), batch `bs` studies, one forward per batch per checkpoint, average logits across
   checkpoints (rank-average across checkpoints via `infer.rank_average` if >1), sigmoid →
   write. Load checkpoints one at a time? No — with ≤3 ViT-S checkpoints keep all resident
   (3 × 88 MB) and run them on the same batch so the decode is shared.
3. Print: ms/study decode, ms/study model, total seconds, projected seconds for 1,300
   studies, peak RSS. Overwrite `submission.csv` atomically at the end AND every 200 studies.

`scripts/smoke_slotknee.sh`: end-to-end on the laptop in < 15 min and < 6 GB RSS:
build cache for `--limit 48` train studies at P=224 → train 1 fold, 2 epochs, bs 4, on those
48 (regex-fallback labels if no LLM CSV) → infer on `data_subset/test_images` → assert
`submission.csv` has the sample's rows/columns and values in (0,1). Prints the three timing
lines. `tests/test_slotknee_pipeline.py` runs the same with `--limit 6`, 1 epoch, P=224,
and must finish in < 90 s on CPU.

## 7. Efficiency benchmark & review (module R)

`scripts/bench_slotknee.py`: measures on THIS machine (MPS, and CPU for reference), per
study and projected for 1,300 studies: (a) header pass, (b) decode of the selected ~27
slices, (c) encoder forward for `6·G` triplets at P ∈ {224, 252}, fp32 vs fp16 (MPS
supports fp16 inference), encoder_chunk ∈ {16, 64}, G ∈ {1, 2, 3}; (d) RSS and MPS
allocated memory. Also benchmark alternative encoders *for the record only* (timm
`vit_tiny_patch16_224.augreg_in21k`, `efficientnet_b0`, `convnext_nano`, `mobilenetv3_small`)
so the review can state what a smaller student would buy in seconds, using the J formula.
Writes a table to `docs/slotknee_review.md` together with a design review of the spec and of
the four modules as they land: concrete, numbered, ranked by expected ΔJ, each with the
evidence for it. Do not edit `src/`.

## 8. Reporting

Each module's build notes: files written, how to run the tests, measured numbers (time,
RSS), any interface deviation from this spec, and open risks. Keep it under 40 lines.
