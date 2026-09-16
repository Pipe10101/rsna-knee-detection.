# Deep-research round 2 (2026-08-24): performance-per-second candidates

Constraints these were selected against: T4, ≤9 h, efficiency exchange rate 0.01 AUC ≈ 717 s,
weak LLM labels (noisy by construction), g10t1 coverage input (60×224² imgs/study),
champion = 5-fold DINOv2-S ensemble, OOF 0.8531 / LB 0.866. Every candidate enters through
the standard gate (2 seeds where feasible, pooled OOF, ±2×noise). One delta per arm.

## Tier 1 — cheap, high-confidence, schedule now

### 1. Fast DICOM decode (efficiency, biggest single win available)
Submit-kernel measurement: decode 3607 ms/study vs model 1176 ms — decoding is ~70% of
runtime. Kaggle RSNA-competition standard fixes: `dicomsdl` (multi-× faster than pydicom on
CPU), and GPU decode of JPEG2000 via nvJPEG2000 / nvImageCodec (5×; DALI route up to 17×);
nvImageCodec ships a pydicom plugin that accelerates `.pixel_array` transparently.
Effect: submit runtime 85 → ~35–45 min ⇒ ~0.04–0.06 AUC-equivalents on the efficiency
track, and free headroom for more TTA/members on the main track.
Implementation: wheels for dicomsdl (+ nvidia-nvimgcodec-cu12) into the code dataset;
decode-pool switch in `scripts/infer_slotknee.py` behind `--decoder {pydicom,dicomsdl,gpu}`.
Gate: pixel-level parity on the 649 local studies (|Δ|=0 required for lossless syntaxes),
then runtime A/B in a CPU staging run. No AUC risk if parity holds.

### 2. reg4 backbone (in flight — fold 0: +0.014 pooled, ACL +0.042)
Mechanism is documented, not folklore: DINOv2 ViTs produce high-norm artifact tokens that
pollute patch-token attention (ICLR'24 "ViTs need registers"); registers absorb them. Our
head attends over patch tokens per finding ⇒ direct beneficiary. If pooled fold 0+1 gates
ADOPT: retrain final with `--backbone vit_small_patch14_reg4_dinov2.lvd142m` (weights
already in the code dataset).

### 3. Per-label deep AUC maximisation finetune (`aucm` arm)
Our metric IS macro AUC; BCE is a proxy. LibAUC's AUC-margin loss won CheXpert (#1 of 150+,
+2% over baseline) and top-1% Melanoma. Best practice: normal BCE training, then last 2–3
epochs switch to AUCM per label (compositional/alternating). Implementation: `--loss aucm
--aucm-epochs 3` in train_slotknee (12 independent binary AUCM losses, masked). Laptop arm.

### 4. Sharpness-aware minimisation on the trainable part (`sam` arm)
SAM variants show consistent generalisation gains specifically under label noise (our
regime). Cost: 2× fwd/bwd of head+4 blocks only (frozen prefix excluded) ≈ +35% step time.
Implementation: SAM wrapper on param_groups, rho 0.05, head-only first. Laptop arm.

## Tier 2 — structural, next T4/laptop slots

### 5. DINOv3 gate (`dinov3` arm) — released 2025-08-13, permissive license, in timm
Radiology transfer evidence (chest-radiograph study, arXiv 2510.07191): at 224 px DINOv3 ≈
DINOv2; it pulls ahead only at 512 px on small/boundary findings, and ConvNeXt-DINOv3
beats ViT-B/16 there. Read for us: (a) vits16 at 224 is a *free efficiency* candidate —
196 tokens vs our 256 (~25% faster) at likely-equal AUC; (b) the AUC play is
`convnext_small.dinov3_lvd1689m` at 384–448 **on the zoom slots only** (small structures:
ACL notch, meniscus roots) — resolution×zoom×CNN, untested by our p336-null (that was G4
low coverage, ViT, image-centred). Weights → code dataset first.

### 6. Token Merging (ToMe) at inference only
Training-free ~2× ViT throughput with minimal accuracy loss (r≈8–13 per layer). Stacks
with candidate 1 for the efficiency entry. Gate: apply to champion at inference, re-score
OOF — accept only if ΔAUC ≥ −0.001. Implementation: timm-compatible ToMe patch in
infer_slotknee behind `--tome-r`.

### 7. Distill2 → noisy-student loop (queued; teacher_g10_oof.csv ready)
Old (leaked) distill showed +0.0127 — the only big head-knob positive. Clean protocol per
Noisy Student: soft targets, student sees augmentations (once cutrot gates), teacher
targets stay clean-input. If distill2 gates: iterate teacher←student once (self-training),
which is also the efficiency-entry pipeline.

## Tier 3 — informed deprioritisation (evidence says don't spend here)
- **Soups**: ensembles ≥ soups in-domain (soups win only under distribution shift); our
  hidden set is same-domain. Keep soup only as an efficiency fallback measurement.
- **Learned MIL/attention slice-aggregation**: 2026 multi-dataset benchmark: mean pooling
  matches or beats attention-MIL in most 3D-neuroimaging tasks (gains ≤0.025 ever, usually
  ~0). Keep our mil/seqmix arms cheap and expect NULL; do not extend this family further.
- **Bigger encoders at 224** (ViT-B/L, any family): three independent nulls now (ours ×2,
  forum ×1) + chest-radiograph study (no DINOv3-at-224 win). Only the vitb gate arm remains.

## Sources
- nvImageCodec DICOM decode: docs.nvidia.com/cuda/nvimagecodec/samples/DICOM-pydicom.html
- RSNA mammo decode threads (5×/17×): kaggle.com/competitions/rsna-breast-cancer-detection/discussion/374248
- ViTs need registers (ICLR'24): arxiv.org/abs/2309.16588
- LibAUC / AUCM CheXpert+Melanoma: arxiv.org/abs/2012.03173, libauc.org
- SAM under label noise: arxiv.org/abs/2411.17132, NCSAM arxiv.org/abs/2601.19947
- ToMe: arxiv.org/abs/2210.09461, github.com/facebookresearch/ToMe
- DINOv3: arxiv.org/abs/2508.10104, github.com/facebookresearch/dinov3
- DINOv3 radiology resolution study: arxiv.org/abs/2510.07191
- MIL benchmark (mean-pool parity): arxiv.org/abs/2604.26807
- Model soups in/out-of-domain: arxiv.org/abs/2203.05482
- Noisy Student: arxiv.org/abs/1911.04252

## Implemented 2026-08-24 (main) — arm lines for the NEXT driver restart

`--loss aucm` / `--aucm-epochs N` / `--aucm-margin` (src/train.py `MaskedAUCMLoss`) and
`--sam-rho` (first-order SAM over trainable params) are in `scripts/train_slotknee.py`,
default-off and verified bit-identical on the default path (tests/test_aucm_sam.py, 8 tests;
suite 466 green; CPU e2e smoke: BCE epoch → AUCM epoch → checkpoint).

Append to `scripts/ablate_slotknee.sh` at the next restart ONLY (never mid-run — bash
re-reads a running script by byte offset):

```sh
    run_arm distill2  "$SEED" --distill-targets data_subset/labels_external/teacher_g10_oof.csv --distill-weight 0.5
    run_arm aucm      "$SEED" --aucm-epochs 3
    run_arm sam       "$SEED" --sam-rho 0.05
```

Notes: `--sam-rho` needs a scaler-free AMP mode (laptop MPS/bf16 is fine; a T4 fp16 arm
must pass `--amp bf16` or it exits with a clear message). AUCM is a final-phase loss: with
bs 8 a rare finding often has no pos/neg pair in a batch, so from-scratch `--loss aucm` is
expected to underperform `--aucm-epochs 3` — gate the latter first.

## DINOv3 readiness (scout, 2026-08-24)

Verified against the installed timm 1.0.28 / torch 2.8.0 on CPU. Item 5 above is now
runnable for the ViT and blocked for the ConvNeXt.

**Exact timm names** (`timm.list_models('*dinov3*', pretrained=True)`, 21 entries):

| candidate | timm name | tokens @224 | in SlotKneeS |
|---|---|---|---|
| DINOv3 ViT-S/16 | `vit_small_patch16_dinov3.lvd1689m` | 201 = 196 patch + 5 prefix (1 CLS + 4 storage) | **yes**, after today's fix |
| DINOv3 ViT-B/16 | `vit_base_patch16_dinov3.lvd1689m` | 201, embed 768 | yes (same code path) |
| ConvNeXt-S DINOv3 | `convnext_small.dinov3_lvd1689m` | none — 7×7×768 feature map | **no** |

Also present: `vit_small_plus_patch16_dinov3`, the `*_qkvb` variants (extra qkv bias, same
geometry), `vit_{large,huge_plus,7b}_patch16_dinov3`, `convnext_{tiny,base,large}.dinov3_*`.
The `.sat493m` tags are satellite-pretrained — ignore them.

**Token counts vs today.** Champion `vit_small_patch14_dinov2` = 257 (256+1); the newly
adopted reg4 = 261 (256+5); DINOv3 ViT-S = **201 (196+5)**, i.e. 0.77× the tokens. Measured
CPU forward, 12 images through `encode_images`: 263 ms (dinov2 S/14) → 202 ms (dinov3 S/16),
−23%. Bonus: DINOv3 has **no `pos_embed` parameter at all** (position is pure RoPE), so
running it at 224 involves zero position-embedding resampling — unlike DINOv2, which is
pretrained at 518 and interpolated down.

**What broke and what was fixed (`src/slotknee.py`, ~25 lines, default-off by construction).**
timm implements the DINOv3 ViTs with the `Eva` class, not `VisionTransformer`, and
`SlotKneeS` drives the block stack by hand. Three real breakages:

1. `Eva.patch_drop` is `None` (not `Identity`) → `__init__` died with
   `AttributeError: 'NoneType' object has no attribute 'parameters'` while freezing.
2. `Eva._pos_embed(x)` returns a **tuple** `(tokens, rope)`, so `enc.patch_drop(t)` /
   `enc.norm_pre(t)` got a tuple.
3. Silent, and the dangerous one: `Eva` blocks take `rope=` and **do not error without it**.
   Running the 12 blocks with `blk(t)` gives cos 0.887 / max|Δ| 2.28 against the correct
   output — a positionless encoder that would have quietly scored as a NULL arm.

Fix: filter `None` out of `_frozen_modules`; unpack the `_pos_embed` tuple; new `_rope()`
helper that rebuilds the table from `encoder.rope.get_embed(shape=(P/patch, P/patch))` and
caches it (bitwise equal to what `Eva._pos_embed` computes, and it also lets the
cached-activation tail path reconstruct rope, which it otherwise cannot); pass `rope=` in
both the frozen and the trainable loop, with per-block `torch.utils.checkpoint` in place of
`checkpoint_seq` (which forwards no kwargs) when grad checkpointing is on.
Verified: SlotKneeS prefix+tail CLS == `enc.forward_features` CLS to **0.0**; forward +
grad-checkpointed backward finite, 76/76 trainable tensors get grads; `param_groups`
correct (blocks.8–11 LLRD, norm, head). DINOv2 numerics are unchanged by construction
(`rope is None` on every new branch); `tests/test_slotknee.py` 45 passed,
`tests/test_slotknee_pipeline.py` 10 passed.

**ConvNeXt-S: not feasible without a new encoder adapter — do not schedule it as an arm.**
It fails before `param_groups` is ever reached:
`timm.create_model(..., img_size=224)` → `TypeError: unexpected keyword argument 'img_size'`
(ConvNeXt has no fixed input size); with that dropped it fails the
`_REQUIRED_ENCODER_ATTRS` guard — missing `patch_embed`, `_pos_embed`, `patch_drop`,
`blocks`, `norm` (it has `stem` + `stages` of depth 3/3/27/3, and `embed_dim` is `None`).
And confirming the brief's suspicion: `param_groups()` iterates `self.encoder.blocks[i]`
and `self.encoder.norm`, both absent, so it would raise `AttributeError` even if
construction were forced. Minimal fix ≈ a second encoder class: stage-based freezing via
`src.model.depth_container` (which already returns `("stages", ...)` for ConvNeXt), a
per-stage LLRD branch in `param_groups`, and a `[B, 768, 7, 7]` → token-axis flatten in
`_pool_tokens`. That is a genuine refactor, not a patch — and item 5's ConvNeXt play only
pays at 384–448 on zoom slots, which also needs a new cache. Deferred, weights **not**
staged (would be 197.9 MB of dead payload).

**Weights staged.** `kaggle/slotknee_code/weights/vit_small_patch16_dinov3.lvd1689m.safetensors`,
**86.4 MB** (from `timm/vit_small_patch16_dinov3.lvd1689m`, ungated). Directory total
522.8 MB → **609.2 MB** (499 → 581 MiB). Verified offline (`HF_HUB_OFFLINE=1`) through the
normal `pretrained_path` route for both `T=1` and `T=3`; `_resolve_weights_file`
stem-matches it unambiguously against the three DINOv2 files already there. Not staged:
ViT-B/16 DINOv3 (342.6 MB — only worth it if `dinov3s` gates ADOPT; it does instantiate in
SlotKneeS today, embed 768 / depth 12 / 201 tokens) and ConvNeXt-S (197.9 MB, see above).
Licence caveat for the submission writeup: the DINOv3 weights ship under Meta's
**"dinov3-license"**, not Apache-2.0 like DINOv2 — check it against the competition's
external-data rules before a DINOv3 model goes into a final submission.

**Arm lines** — append to `scripts/ablate_slotknee.sh` at the next restart only (same
byte-offset caveat as above):

```sh
    run_arm dinov3s   "$SEED" --backbone vit_small_patch16_dinov3.lvd1689m --pretrained-path kaggle/slotknee_code/weights
```

and, only if `dinov3s` gates ADOPT, the capacity follow-up (stage its 342.6 MB of weights first):

```sh
    run_arm dinov3b   "$SEED" --backbone vit_base_patch16_dinov3.lvd1689m --pretrained-path kaggle/slotknee_code/weights
```

Gate notes: this is a *two-sided* arm — a pooled-OOF delta inside ±2×noise is still a win
because it buys ~23% encoder time, so record wall-clock per epoch alongside the AUC.
It runs on the existing g10t1 **image** cache unchanged (224 is a clean multiple of 16),
but any `--use-cached-activations` file must be rebuilt: the frozen prefix and the token
count both change (257 → 201), and `check_cached_activations` correctly refuses the old
file with a backbone mismatch. Not yet done: no T4/laptop run, no OOF — this is
instantiation-and-weights readiness only.

## ⚠️ CORRECTION (2026-08-24 20:00) — read this BEFORE the section below

The section that follows scores the model and the LLM labels against the 58 gold studies and
concludes which findings are "model-limited". **Its yardstick is broken.** An independent
re-extraction of all 58 gold reports agrees with the public LLM labels at **0.999 macro** — two
independent readers, near-identical — yet both score only ~0.89 against gold. The re-extraction
contradicts gold in **76/696 cells (10.9%)**, and the public labels side with it against gold in
**74 of 76**. Gold's errors are therefore broad, not confined to Lateral OA and Synovitis.

So "labels 0.909" is not label quality — it is the report-to-gold AGREEMENT ceiling, with gold
as the unreliable side. **Do not use the per-label gaps below to rank label sets, to fit blend
weights, or to decide where the model is weak.**

What survives, because it never used gold: the pooled OOF over all 4,407 studies against the LLM
labels — the metric that has tracked the public LB ~1:1 three times. On that metric the weak
findings are **Lateral Meniscus 0.748, MCL 0.778, ACL 0.806**, which is the same conclusion the
section below reaches by a worse route. The coronal-zoom rationale therefore stands; the
"label-limited vs model-limited" split does not.

## The decisive measurement (main, 2026-08-24 14:30): which labels are model-limited

Scored the champion's 5-fold OOF against the 58 **true radiologist-labelled** studies, and
against the same gold, the LLM training labels themselves. Model macro **0.837**, LLM labels
**0.893** → the label ceiling is only **+0.056** away, and it is NOT evenly distributed:

| finding | model vs gold | labels vs gold | gap | reading |
|---|---|---|---|---|
| MCL | 0.673 | 0.968 | **−0.295** | model-limited |
| Lateral Meniscus | 0.648 | 0.879 | **−0.230** | model-limited |
| Medial Meniscus | 0.816 | 0.948 | −0.132 | model-limited |
| PF OA | 0.788 | 0.902 | −0.114 | model-limited |
| ACL | 0.878 | 0.987 | −0.109 | model-limited |
| Lateral OA | 0.743 | 0.833 | −0.090 | model-limited |
| Synovitis | 0.786 | 0.790 | −0.004 | at the ceiling |
| Medial OA | 0.944 | 0.932 | +0.012 | **beats its labels** |
| Baker's | 0.964 | 0.944 | +0.020 | **beats its labels** |
| Contusion | 0.933 | 0.860 | +0.073 | **beats its labels** |
| Effusion | 0.963 | 0.877 | +0.086 | **beats its labels** |
| Fracture | 0.903 | 0.793 | +0.110 | **beats its labels** |

CAVEAT, sharpened (cross-checked against the campaign log): the gold-58 set has
*verified* label errors concentrated in **Lateral OA and Synovitis** — exactly two rows above.
Strike both: LatOA's −0.090 may be a gold artefact rather than a vision deficit, and the
Synovitis "ceiling" reading is not evidence of anything. On the 10 trusted labels the picture
is unchanged and slightly starker: **model 0.851 vs labels 0.909, headroom +0.058**. The robust
model-limited set is **MCL, LatMen, MedMen, PF OA, ACL** — and every non-ACL member is a
coronal joint-line structure, so the zoomc case rests on the cells that survive the caveat, not
the shaky ones. Also note 58 studies means wide CIs (MCL rests on 9 positives). Directionally, though, two conclusions are hard to escape:

**1. The remaining headroom is an IMAGE-READING problem in six findings, not a label problem.**
MCL + lateral meniscus alone are over half the total gap, and both are **coronal joint-line**
structures — while our only zoom view is a *sagittal* ACL notch. Hence `cache_builder_zoomc`
(coronal joint-line zoom, ~80 mm, on the g10t1 layout): it targets MCL, both menisci and both
OA compartments at once — four of the six model-limited findings. Highest-value untried INPUT.
Note this also explains why generic capacity arms (ViT-B, tb12, more epochs, higher lr) keep
measuring NULL: the bottleneck is *where the model looks*, not how much it can memorise —
exactly the pattern that made coverage (+0.0325) and reg4 (+0.0220) the only big winners.

**2. Distillation should be applied PER LABEL, not uniformly.** On five findings the model is
already a better teacher than its own targets (Fracture +0.110, Effusion +0.086, Contusion
+0.073), while on MCL/LatMen the labels are far better than the model — so a uniform
`--distill-weight 0.5` simultaneously helps the first group and *corrupts* the second.
FOLLOW-UP ONLY (one delta per experiment): gate plain `distill2` first; if it ADOPTs, add
`--distill-weights-json` (per-label weights ∝ the gap above, ~0.7 where the model wins, ~0.1
where the labels win) as its own arm. Not implemented yet — deliberately, until distill2 reports.

## CORRECTION (2026-08-24 15:00): "decode is 70% of runtime" was WRONG — measured on Kaggle

`slotknee-decode-bench` (CPU kernel, 5.4 min) settled it against the real competition data:

- **Nothing is compressed.** 557/557 visible TEST files and 2214/2214 sampled TRAIN files are
  `Explicit VR Little Endian`. There is no JPEG2000/JPEG-Lossless anywhere in the data we can
  see, so the 5×-decoder premise (borrowed from the RSNA mammography competitions) does not
  apply here. The pylibjpeg decoders stay installed as insurance, but they are not the lever.
- **The submit kernel's "3607 ms/study decode" is not decode.** It is `info["ms"]` = the whole
  build (header index + pixel decode) summed over 3 anchor shifts. Real pixel decode is ~13%
  of a cold study (307 ms) and the header walk is ~87% (2113 ms).
- **dicomsdl is therefore worth ~6.6%**, not 60%: ~78 → ~73 min. Ship it anyway — parity is
  proven (20/20 test slices and 43/43 study tensors bit-identical on Kaggle, 0 fallbacks) and
  it can only ever fall back to pydicom — but bank ~5 min, not 40.

**The real lever was next door, and is now fixed.** `scripts/infer_slotknee.py::_decode_study`
called `build_study_tensor` once per anchor shift *without* passing `index=`, so the expensive,
anchor-shift-INDEPENDENT header walk was redone for every TTA shift. It now indexes once and
reuses it: verified byte-identical tensors for shifts 0/+1/−1 on real studies, **186 → 123 ms
per 3-shift study locally (1.51×, 34% saved)**; on Kaggle's cold I/O the saving should be
larger still, since that is where the header walk costs 2113 ms.

Standing lesson: profile the claim before optimising it. The 70% figure came from reading a
log line's label instead of its definition, and it would have bought a ~5-minute win at the
price of a wheel, a flag and a week of attention. Retired.

---

## Built (2026-08-24, cache engineer): `slots_P224_g10t1_zc` — the coronal joint-line zoom

**Motivation** — the measurement above. Four of the six model-limited findings (MCL 0.295,
Lateral Meniscus 0.230, Medial Meniscus 0.132, Lateral OA 0.090, and PF OA 0.114 partly) are
read on **coronal** images at the tibiofemoral joint line, while the only zoom view built so
far is the *sagittal* ACL-notch one (`slots_P224_g10t1_z100j`). This is an INPUT change, the
class of change that produced the only two large winners so far (coverage +0.0325, reg4
+0.0220), rather than more capacity, which keeps measuring NULL.

**What it is** — `kaggle/cache_builder_zoomc/` (CPU kernel `felipedeleon11/slotknee-cache-zoomc`,
registered as `cachezoomc` in `kaggle/kaggle_ops.py`; push with
`python3 kaggle/kaggle_ops.py push-cache-zoomc`). The g10t1 coverage layout (P=224, G=10, T=1)
plus two spec-zoom slots via `build_slot_cache.py --zoom-spec "COR_FS:80:joint,COR_T1:80:joint"`:

| | |
|---|---|
| cache dir | `slots_P224_g10t1_zc` |
| slot names | `SAG_FS, COR_FS, AX_FS, SAG_T1, COR_T1, AX_T1, COR_FS_Z80J, COR_T1_Z80J` |
| S / row | 8 / `[8, 10, 1, 224, 224]` = 4.01 MB |
| size | 4,407 × 8 × 10 × 224² = **16.5 GiB** (under the 20 GB working cap; same footprint as zoomj, which completed in one session) |
| zoom crops | 80 mm → 0.36 mm/px at P=224, centred on `src/slots.locate_joint`; cut from the SAME single decode as the base 140 mm crops |

Override `SLOT_ZOOM_SPEC="COR_FS:80:joint"` for the COR_FS-only variant (S=7, 14.4 GiB) if the
T1 fallback rate below proves harmful. `SLOT_SKIP` / `SLOT_LIMIT_N` shard it like `g20a/g20b`.

**Coronal locator health, measured on 40 real studies BEFORE the push** (the zoom would be
worthless if it silently degenerated to the image centre):

| slot | present | fell back to image centre | median conf | median joint offset |
|---|---|---|---|---|
| `SAG_FS` (what zoomj already ships on) | 39/40 | 3 (8%) | 0.65 | +13.1 mm |
| **`COR_FS`** | 40/40 | **5 (12%)** | 0.64 | **+13.7 mm**, \|dy\|≥5 mm in 33/35 |
| `COR_T1` | 34/40 | **15 (44%)** — "no bone masses"; bright-marrow polarity is weak on T1 | 0.89 | +16.0 mm |

So the coronal FS locator works at the same rate as the sagittal one already in production.
Verified end-to-end through the real CLI on 3 studies (`--zoom-spec`, read back with
`SlotCache`): S=8, `zoom_spec` recorded in the index, and the zoom slot is demonstrably NOT a
duplicate of the base slot — crop windows move 4–81 rows / 0–57 cols, pixel means differ
(e.g. COR_FS 39.6 vs COR_FS_Z80J 48.4) and base↔zoom correlation is 0.11–0.38. Note a
fallback does **not** degenerate to the base crop: it is still an 80 mm zoom, image-centred.

**Gate line (do not run yet — one delta per experiment, and the arm queue is owned elsewhere).**
Kaggle, `kaggle/ablate_kernel_g10/slotknee-ablate-g10.py` `ARMS`, plus
`"felipedeleon11/slotknee-cache-zoomc"` in that kernel's `kernel_sources`:

```python
    ("zoomc",      [], "slots_P224_g10t1_zc"),   # coronal joint-line zoom: MCL + both menisci + both OA compartments
```

Laptop, `scripts/ablate_slotknee.sh`:

```bash
    if [ -d "${CACHEZC:-/nonexistent}" ]; then ARM_CACHE="$CACHEZC" run_arm zoomc "$SEED"; fi
```

Gate as usual (2 seeds, pooled OOF, ±2×noise) but read the **per-finding** deltas on MCL,
Lateral/Medial Meniscus and Lateral/PF OA, not only the macro: this arm is aimed at four
labels out of twelve, so a real win can hide inside a flat macro.

## Built (2026-08-24, cache engineer): `slots_P336_g10t1` — resolution ON TOP of coverage

**Motivation — the one combination never tested.** Two things are measured, not assumed:

1. **Input size is what wins.** Going from 18 to 60 images/study (the `g10t1` coverage layout)
   gave **+0.0325 pooled OOF**, the biggest win of the campaign, while every capacity arm
   (ViT-B ×2, `tb12`, `ep14`, `lrb1e4`) measured NULL.
2. **Resolution at the COST of coverage loses.** The existing `slots_P336_g4t1` cache is P=336
   with only G=4 anchors (24 imgs/study, chosen for equal compute) and measured **−0.0070** —
   a NULL/slight regress.

Nobody has yet run resolution **in addition to** coverage: P=336 at G=10 = 60 imgs/study at
24×24 patches = **2.25× the tokens** of the current champion. Felipe has explicitly authorised
trading inference speed for accuracy and the budget supports it (the current submission uses
~60 min of a 9 h limit). It matters because the remaining weakness is concentrated in **small
structures** — against true radiologist labels MCL 0.673, lateral meniscus 0.648, medial
meniscus 0.816, PF OA 0.788 — and the published radiology-transfer evidence says resolution is
precisely what helps small / boundary findings.

### Size check (done BEFORE the push — this is why it ships as two halves)

`N × S × G × T × P × P` uint8, N=4,407, S=6, G=10, T=1:

| cache | bytes | GB (10⁹) | GiB (2³⁰) | fits Kaggle's 20 GB output cap? |
|---|---|---|---|---|
| `slots_P224_g10t1` (champion, built) | 13,267,537,920 | 13.27 | 12.36 | yes |
| `slots_P336_g4t1` (built) | 11,940,784,128 | 11.94 | 11.12 | yes |
| **`slots_P336_g10t1` (this, FULL)** | **29,851,960,320** | **29.85** | **27.80** | **NO — 1.49× over** |
| `slots_P336_g10t1_partA` (2,204 studies) | 14,929,367,040 | 14.93 | 13.90 | yes |
| `slots_P336_g10t1_partB` (2,203 studies) | 14,922,593,280 | 14.92 | 13.90 | yes |

**Verdict: the full cache cannot be built by one kernel.** The laptop is worse off still —
`df -h` reports **18 GiB free** (11 GB `slots_P224_full` + 12 GB `slots_P224_g10t1_full`
already resident), so the merged 27.80 GiB has no local home and a local merge (parts + merged
= 55.6 GiB) is impossible on this machine. It does not need one: `kaggle/ablate_kernel_g10`
consumes caches through `kernel_sources`, so both halves attach directly on Kaggle and never
touch the laptop. Cheapest fix, and the one implemented: **shard into halves exactly like
`cache_builder_g20a`/`g20b`** — same sorted uid list, part A `--limit 2204`, part B
`--skip 2204`, merged by `scripts/merge_slot_cache.py` (verified end-to-end below).

### What was built

`kaggle/cache_builder_p336g10a/` and `kaggle/cache_builder_p336g10b/` — CPU script kernels
`felipedeleon11/slotknee-cache-p336g10a` / `-p336g10b` (`"enable_gpu": "false"`), registered in
`kaggle/kaggle_ops.py` as `cachep336g10a` / `cachep336g10b` with `cachep336g10` as an alias for
part A; push with `push-cache-p336g10a` / `push-cache-p336g10b` (both in the argparse **choices
list AND the dispatch dict** — `push-cache336` once shipped in the dict only and was
uninvokable). Structure copied verbatim from `cache_builder_g10t1` / `g20a`: the nested
`/kaggle/input` walk with the predicate tested **before** pruning the 800k-file image trees,
`kaggle_bootstrap.py` symlink data_dir, uint8 memmap shard writes, `--limit-minutes 500`
resumability, part A alone building the 3-study test split.

| | |
|---|---|
| cache dirs | `slots_P336_g10t1_partA`, `slots_P336_g10t1_partB` → merged `slots_P336_g10t1` |
| row | `[6, 10, 1, 336, 336]` uint8 = **6.77 MB/study** (vs 3.01 MB at P=224) |
| shard size | **256** studies (not the template's 512): 512 rows would be a 3.47 GB shard file, larger than anything this pipeline has moved; 256 gives 1.73 GB, inside the proven 1.4–1.5 GB range, and halves the cost of a resume |
| everything else | identical to `g10t1`: crop 140 mm, trim 0.15, laterality mirroring, no zoom slots |

### Measured runtime multiplier (real number, not an assumption)

`build_study_tensor(P=336, G=10, T=1)` vs `P=224` on 4 real studies from
`data_subset/train_images`, both sizes pre-warmed and interleaved so neither pays the cold
page-cache read:

| study | slots filled | t P=224 | t P=336 | ×  |
|---|---|---|---|---|
| …4356622764 | 6/6 | 0.073 s | 0.098 s | 1.34 |
| …1304016339 | 6/6 | 0.078 s | 0.102 s | 1.31 |
| …8156266866 | 4/6 | 0.055 s | 0.063 s | 1.14 |
| …7366582736 | 4/6 | 0.053 s | 0.060 s | 1.14 |
| **total** | | 0.259 s | 0.323 s | **×1.25** |

Shape/dtype confirmed `(6, 10, 1, 336, 336)` `uint8`, 65–97 % non-zero pixels, masks 4–6/6.
**Build cost is dominated by DICOM decode, not by the resample to P** — the same 60 slices are
decoded either way, so CPU time grows only ×1.25 while write volume grows ×2.25. Against the
`slots_P224_g10t1` Kaggle report (4,407 studies, 4 workers, **30.4 min**, 1.49 s/study), a half
at P=336 projects to **≈ 2204 × 1.49 × 1.25 / 4 ≈ 17 min** — the 20 GB cap, not the clock, is
the binding constraint. (A naive cold-cache measurement reads ×0.78, i.e. *faster* at 336; that
is the page cache warming between the two calls, which is why the table above interleaves.)

End-to-end proof through the real CLI, not just the library: a 3-study part A (`--limit 3`) and
a 3-study part B (`--skip 3 --limit 3`) were built at P=336/G=10/T=1/`--shard-size 256`
(`train_x.u8` = 20,321,280 B = 3 × 6,773,760 ✓), merged with `scripts/merge_slot_cache.py`
(6 studies, 2 shards), and read back through `src.slots.SlotCache`: `S=6 G=10 T=1 P=336`,
row shape `(6, 10, 1, 336, 336)`, mask `[1,1,1,1,1,1]`.

### Push status

Both pushed as **version 1** and confirmed `RUNNING` (CPU sessions; no GPU kernel was pushed).

### Gate line for the ablate-g10 kernel (2 folds) — do NOT run until the arm queue owner schedules it

The two halves must be merged first. Add **both** kernels to
`kaggle/ablate_kernel_g10/kernel-metadata.json` `kernel_sources`:

```json
    "felipedeleon11/slotknee-cache-p336g10a",
    "felipedeleon11/slotknee-cache-p336g10b",
```

then in `kaggle/ablate_kernel_g10/slotknee-ablate-g10.py` `ARMS`:

```python
    ("p336g10",    [], "slots_P336_g10t1"),   # resolution ON TOP of coverage: 60 imgs/study @336 = 2.25x tokens
```

with `FOLDS = "0 1"` (2 folds, 8 epochs, seed 42 — the same reference as every other arm on
this layout). Two caveats for whoever wires it:

* The arm loop looks up `cache_name` in the dict of attached `slots_*` dirs, and neither half
  is named `slots_P336_g10t1`. The kernel must first run
  `scripts/merge_slot_cache.py --parts <partA> <partB> --out $TMP/slots_P336_g10t1` — the
  parts live on the read-only `/kaggle/input` mount, so `--link` (hardlink) will fall back to
  copy and move 27.8 GiB into `/kaggle/temp`. Cheapest follow-up if that is tight: give
  `merge_slot_cache.py` a `--symlink` mode (index rewrite only, zero bytes copied) — the merged
  cache is read-only for training anyway, which is exactly what that script already documents.
* Budget ×2.25 tokens per step. Time the arm against the `g10t1` reference (~2.5 h for 2 folds)
  before assuming three arms still fit one 9 h session; `p336g10` alone may need most of it.

Read the **per-finding** deltas on MCL and the menisci alongside the macro: the whole premise
is that resolution buys small structures, so a real win can hide inside a flat macro.

## Big-input feasibility, measured (main, 2026-08-24 17:45)

The two levers that could move the campaign median (336 px at full coverage; all-slices G20)
are worthless if they OOM or time out on a T4, so both were profiled before any session is
spent. Isolated processes (`ru_maxrss` is a per-process high-water mark — measuring several
configs in one process silently under-reports every one after the first).

| config | peak RSS | model+activations | s/step (CPU) |
|---|---|---|---|
| P=224 G=10 bs=2 (champion) | 1.52 G | 1.26 G | 4.7 |
| P=336 G=10 bs=2 | 2.68 G | 2.41 G | 12.4 |
| P=224 G=20 bs=2 | 1.75 G | 1.48 G | 9.9 |
| P=336 G=10 bs=4 | 2.93 G | 2.66 G | 28.1 |
| P=224 G=10 bs=8 (production) | 2.20 G | 1.93 G | 23.2 |

**Memory: not a blocker.** Doubling the batch from 2 to 4 at 336 px adds only 0.25 G, because
`encoder_chunk=64` + gradient checkpointing bound activation memory independently of batch.
Extrapolating to production bs 8 gives ~3.2 G against the T4's 15.8 G — comfortable. Neither
experiment needs a batch-size reduction, so neither confounds its gate with a different
effective learning rate.

**Time is the binding constraint.** Per image, 336 px costs ~2.4x and G20 ~2.1x. A 2-fold
8-epoch arm at 224/G10 takes ~125 min on a T4, so:
- **G20 gate: ~260 min** — fits a session with room for an in-session control.
- **336/G10 gate: ~300-390 min** — fits a session, but ALONE. `ablate_kernel_g10` runs its
  ARMS list in order with `SK_TOTAL_MINUTES=500`; leaving the usual 4-5 arms in place would
  starve it. Set `ARMS` to that single arm (plus `--limit-minutes`) for that session.

**Latent blocker found and cleared**: `SlotKneeS(max_groups=8)` by default, so G=20 raises
`ValueError: G=20 exceeds max_groups=8`. `scripts/train_slotknee.py` already passes
`max_groups=max(8, cache.G)`, so training is safe — but anything constructing the model
directly (a notebook, an inference path built by hand) must do the same. Inference is safe:
it rebuilds from the checkpoint's `hparams`, which record the trained `max_groups`.

Ready-to-run when the deploy freeze lifts: caches `slots_P336_g10t1` (two halves, merge first)
and `slots_P224_g20` (halves already COMPLETE); one T4 session each; gate at 2 folds against
the reg4 baseline on the same folds.

## Efficiency-aware re-ranking (main, 2026-08-24 18:15) — REVERSES the big-input recommendation

Exchange rate: 0.01 AUC ~ 717 s of runtime over 1300 studies. Applying it to the levers, with
the T4-measured 1.176 s/study of model time (5 members x 3-shift TTA):

**Free AUC — train-time only, +0 s at inference** (this is where the campaign should spend):

| lever | AUC | inference cost |
|---|---|---|
| reg4 backbone (ADOPTED) | +0.022 | 0 |
| distillation (queued) | ~+0.007 | 0 |
| AUCM loss / SAM (queued) | ~+0.004 each | 0 |
| SSL pretraining | ~+0.006 | 0 |
| per-label stack weights | ~+0.004 | 0 |

**Paid AUC — net effect on the efficiency score after the runtime penalty:**

| lever | AUC | runtime penalty | NET |
|---|---|---|---|
| 336 px at full coverage | +0.012 | −0.0267 | **−0.0147** |
| all-slices G20 | +0.008 | −0.0213 | **−0.0133** |
| coronal zoom slots (+2 of 6) | +0.010 | −0.0070 | **+0.0030** |

So **336 px and G20 LOSE on the efficiency score** — the two levers flagged earlier as the way
to move the median are main-track-only, and must not go into the efficiency entry. The coronal
zoom survives because it adds 2 slots (1.33x), not 2.25x. Earlier text recommending 336/G20 as
the campaign's next big step stands ONLY for the accuracy entry.

**Efficiency-positive:** distilling the 5-fold ensemble into ONE student cuts model time
1.176 -> 0.078 s/study (15 forward passes -> 1) = 24 min = **+0.0199 AUC-equivalents of
headroom**. The student wins on the efficiency score if it loses LESS than 0.0199 AUC versus
the ensemble — a bar a well-distilled student normally clears. Add the two shipped runtime
fixes (index reuse + dicomsdl, 104 -> 85 min = +0.0152 equivalents) and DINOv3-S at 224
(196 vs 256 tokens, ~23% less encoder time at ~neutral AUC).

**Where this leaves the distillation pipeline**: its core instinct — distil for the efficiency track —
is right and is now quantified. What is wrong is the teacher. Its plan trains an unproven 86M
ViT-Base ensemble first (~3 T4 sessions; ViT-S->ViT-B measured NULL twice at 224). We already
HAVE a measured teacher: the 5-fold reg4 ensemble, with leak-free targets in
`teacher_g10_oof.csv`. Same endgame, minus the expensive unproven step, and available the
moment a session frees.

**Two entries, two recipes** (the competition allows both):
- ACCURACY entry: ensemble + TTA + every paid lever that gates, runtime irrelevant under 9 h.
- EFFICIENCY entry: ONE distilled reg4 student, no TTA, index reuse + dicomsdl, plus every
  free train-time lever. All the "free AUC" rows above apply to BOTH entries.

## External data & weights: what we may actually use (scout, 2026-08-24)

Settles which of the planned pretraining/backbone experiments are legal at all. Everything
below is READ-ONLY research: no dataset was downloaded, no Kaggle push or submission made
(POLICY 5 respected).

### 0. Provenance — how this was obtained, and how to re-check it

The **Kaggle Python API exposes no rules text**. `competitions_list` gives metadata only,
which does confirm the frame: deadline `2026-10-22 23:59`, reward `77,000 Usd`, category
Research, `is_kernels_submissions_only=True`, `max_daily_submissions=5`, `max_team_size=5`,
`merger_deadline=2026-10-15`, `evaluation_metric=Roc Auc Score`, `user_has_entered=True`
(Felipe has already accepted the rules). There is no rules endpoint on the client and the
internal `api/i/competitions.CompetitionService/GetCompetition` RPC 400s without a browser
session.

**WebFetch and `curl` both return only the SPA shell** (5.6 KB, no state blob) for
`/rules`, `/overview` and `/discussion` — a fetch of these URLs looks like an empty page and
must not be read as "no such rule". Every quotation below was read from the **rendered** page
in the browser pane on 2026-08-24. Anyone re-checking must render, not fetch.

### 1. The competition's own wording

**Code Requirements (Overview tab) — the clearest single line, and it is permissive:**

> "Freely & publicly available external data is allowed, including pre-trained models"

listed alongside "CPU Notebook <= 9 hours run-time", "GPU Notebook <= 9 hours run-time",
"Internet access disabled", "Submission file must be named submission.csv". This confirms our
operating assumptions (9 h, internet off) and puts pretrained models squarely inside the
permitted category — subject to "freely & publicly available".

**Rule 2.6(a), EXTERNAL DATA AND TOOLS** — note it is a *disjunction*, either branch suffices:

> "you will ensure the External Data is either publicly available and equally accessible to
> use by all Participants of the Competition for purposes of the competition at no cost to the
> other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b"

and, importantly for us:

> "The ability to use External Data under this Section does not limit your other obligations
> under these Competition Rules, including but not limited to Section 2.8 (Winners Obligations)."

**Rule 2.6.b — the second branch, and the only place the Host's power to prohibit is stated:**

> "The use of external data and models is acceptable unless specifically prohibited by the Host."

with the Reasonableness Standard aimed at "excessive" cost and "geo restrictions" — use "must
be 'reasonably accessible to all' and of 'minimal cost'". The rules' own worked example: a
small LLM subscription is acceptable, whereas "Purchasing a license to use a proprietary
dataset that exceeds the cost of a prize in the competition would not be considered reasonable."
Note what 2.6.b's test is *about*: cost and geo-accessibility. A free click-through is neither.

**Rule 2.5, WINNER LICENSE — this competition's output licence is itself non-commercial**
(Specific Terms §1.6: "WINNER LICENSE TYPE: CC-BY-NC 4.0"; §1.7: "DATA ACCESS AND USE:
Commercial and Academic Research - MIRA license", http://rsna.org/mira-license). And it
carries the carve-out that decides POLICY 4:

> "In the event that input data or pretrained models with an incompatible license are used to
> generate your winning solution, you do not need to grant an open source license in the
> preceding Section for that data and/or model(s)."

**Rule 3.6.c — the commercial-use/OSI constraint binds CODE, not weights, and is overridden here:**

> "if open source code is used in the model to generate the Submission, then you must only use
> open source code licensed under an Open Source Initiative-approved license ... that in no
> event limits commercial use"

— prefaced "Unless otherwise stated in the Specific Competition Rules above", and §2.5 *does*
state otherwise (CC-BY-NC 4.0). Our own code is MIT and we instantiate every backbone through
**timm (Apache-2.0)**, never through Meta's `dinov3` repo code, so 3.6.c is satisfied on the
code axis independently of any weight licence.

**Rule 2.8(a), WINNERS OBLIGATIONS — the clause that actually constrains our choices:**

> "The model weights should be provided as a public kaggle dataset so it is both publicly
> accessible and linked to the inference/submission code."

The Overview tab adds host-specific deliverables: a short video, "publish a link to your open
sourced code and the weights on the competition forum", and "Share final version of model as
publicly available for open distribution and validation." Rule 3.14.a then warrants that "you
are the sole and exclusive owner and rights holder of the Submission, and you have the right to
make the Submission and grant all required licenses."

**Disclosure requirement**: there is **none** during the competition. No external-data
registration thread, no declaration deadline. Disclosure is deferred to the winners' method
description under 2.8(b). Practical consequence: nothing forces us to declare anything now, and
nothing protects us later — the compliance test happens at prize verification, where 3.9.b lets
the Sponsor either disqualify or demand remediation "including ... the resolution of license
conflicts" within one week.

### 2. Host clarifications on the forum — one answered, four unanswered

**ANSWERED** (pinned, Po-Hao "Howard" Chen, COMPETITION HOST badge, #733965 "Use of
Commercially Hosted LLMs"):

> "Use of commercially hosted LLMs and other external inference services is permitted, provided
> that the service and method of use otherwise comply with the Competition Rules"

> "submitting Competition Data, including report text, to an external LLM or API for inference
> or other computational processing (for example, extracting labels from reports) will not, by
> itself, be considered prohibited PRIVATE SHARING of Competition Data outside the Team."

→ **Our entire LLM-label pipeline is explicitly clear**, and so are the public CC0 label
datasets built the same way. The host does reserve "the right to determine whether a particular
service, model, or configuration is reasonably accessible, is prohibitively costly, or otherwise
creates an unfair competitive advantage."

**UNANSWERED, despite four separate requests** (checked 2026-08-24):

| thread | asks | host reply |
|---|---|---|
| #733652 "Rules clarification: external knee-MRI datasets…" | do free click-through datasets satisfy 2.6(a)? | host replied **only** to the LLM half |
| #735497 "Rule 2.6(a) and registration-gated public datasets (MRNet, OAI, etc.)" | direct yes/no on the category | **0 comments**, 8 days old |
| #734109 "Is the gated KneeCoT dataset permitted…" | gated HF dataset, CC BY-NC 4.0 | **0 comments** |
| #735121 "Are CC-BY-NC pretrained weights compatible with the winners open-licence obligation?" | RadImageNet ResNet-50 (CC-BY-NC-SA-4.0) | **0 comments** |

Only the pinned LLM post carries the COMPETITION HOST badge. Everything else in those threads is
participant opinion — including several confident readings of 2.6(b) — and **is not a ruling**.
Do not cite a participant post as permission.

Net: **the Host has not "specifically prohibited" any knee-MRI corpus under 2.6.b, and has not
blessed the gated ones either.** POLICY 2 therefore stands unchanged on its own terms.

### 3. Dataset audit

Verdicts are against POLICY 2 (no registration-gated external data unless the host explicitly
permits). "SSL?" = contains knee MRI usable for self-supervised pretraining, labels not needed.

| dataset | licence | gated? | SSL? | verdict |
|---|---|---|---|---|
| **KMAR-50K** (Mendeley `10.17632/xw7mrg7ntg.6` + `95w9f5tzz8`) | **CC BY 4.0** | **No** — anonymous "Download All", 9.73 GB pt1 | **yes** — 1,444 multi-parameter knee MRI scans, 1,190 patients, paired artifact/clean | **USABLE** |
| **kneeMRI / Štajduhar** (Zenodo 14789903) | **CC BY-NC-ND 4.0** | **No** — Zenodo `access_right: open`, direct download | yes — 917 sagittal PD-FS knee volumes | **USABLE for measurement; NEEDS-FELIPE-CALL before a prize-eligible run** (the **ND** term, see §5) |
| **MRNet** (Stanford, now Stanford Redivis `4a2c-4cpkzrn2c`) | Stanford MRNet Research Use Agreement: "Permission is granted to view and use the MRNet Dataset without charge for personal, non-commercial research purposes only. Any commercial use, sale, or other monetization is prohibited." | **Yes** — page shows "Apply for access"; per-individual registration required | yes — 1,370 knee MRI exams | **BLOCKED** (NEEDS-HOST-RULING to unblock) |
| **OAI** (Osteoarthritis Initiative, `nda.nih.gov/oai/`) | free, non-commercial research; NDA Data Use Certification | **Yes — hardest gate of the set**: NDA account via eRA Commons/Login.gov/PIV **plus a DUC naming an authorized institutional business official** | yes — bilateral knee MRI, longitudinal | **BLOCKED** (not a click-through; needs an institution) |
| **SKM-TEA** (Stanford AIMI) | Stanford University Dataset Research Use Agreement, non-commercial research | **Yes** — must "create a new account or log into an existing account" before download | yes — 155 quantitative 3D knee MRI, k-space + DICOM | **BLOCKED** |
| **fastMRI / fastMRI+** (NYU) | Dataset Sharing Agreement: "use the fastMRI Dataset for internal research or educational purposes only"; must "Not SELL OR OTHERWISE MONETIZE any portion or all of the fastMRI Dataset" | **Yes** — "By registering for downloads from the fastMRI Dataset, I agree to this Dataset Sharing Agreement" | yes — ~10k clinical knee MRI (k-space + DICOM); fastMRI+ adds annotations | **BLOCKED** |
| **KneeCoT** (HF `YiHui0124/KneeCoT`, raised in #734109) | CC BY-NC 4.0 + hospital ethical-use agreement, no redistribution | **Yes** — gated HF access | yes | **BLOCKED** |
| public **CC0 LLM report-label** Kaggle datasets (`stevenleehans/rsna-knee-llm-report-labels`, `pilkwang/rsna-knee-llm-labels`, `lixin73/…-sol56`, in use today) | **CC0: Public Domain** (confirmed via Kaggle REST `licenseName`) | **No** | n/a (labels, not pixels) | **USABLE** — and now doubly safe given the host's LLM ruling |

**The one real find: KMAR-50K.** CC BY 4.0, no registration, no DUA, ~10 GB, genuinely knee MRI,
multi-view/multi-parameter. It clears branch 1 of 2.6(a) on its face and clears POLICY 2 without
needing a host ruling. Its artifact/clean pairs are also a free source of realistic motion-artifact
augmentation, which our current Albumentations MotionBlur only approximates. Not fetched (deploy
freeze); flagged as the candidate worth Felipe's authorisation if we want external SSL data at all.

### 4. Pretrained-weights audit (POLICY 4)

Gating checked against the HF model API (`gated` field) — this is what decides "equally accessible
to all Participants at no cost", since our route is the timm mirror, not Meta's repo.

| weights | licence | HF gated? | redistributable in our Kaggle dataset? | verdict |
|---|---|---|---|---|
| `timm/vit_small_patch14_dinov2.lvd142m` | **Apache-2.0** | No | yes, unconditionally | **USABLE** |
| `timm/vit_small_patch14_reg4_dinov2.lvd142m` (our adopted reg4 champion) | **Apache-2.0** | No | yes | **USABLE** |
| `timm/vit_base_patch14_dinov2.lvd142m` | **Apache-2.0** | No | yes | **USABLE** |
| `timm/vit_small_patch16_dinov3.lvd1689m` (staged, 86.4 MB) | **DINOv3 License** (`license: other`) | **No** — `gated: False` | **yes, with a condition** (below) | **USABLE with conditions** |
| `timm/vit_base_patch16_dinov3.lvd1689m`, `timm/convnext_small.dinov3_lvd1689m` | DINOv3 License | No | same condition | same |
| `facebook/dinov3-*` (Meta's own repos) | DINOv3 License | **`gated: manual`** | — | **do not use this route** — a manual-approval gate is exactly the accessibility problem; the timm mirror is the compliant path |
| RadImageNet ResNet-50 (raised in #735121) | **CC-BY-NC-SA-4.0** | n/a | share-alike propagates | **NEEDS-FELIPE-CALL** — the **SA** term, not the NC term, is the problem (see §5) |

**DINOv3 licence verdict: NOT the blocker POLICY 4 assumed.** Read from
`facebookresearch/dinov3/LICENSE.md` (Last Updated August 19, 2025). §1.a grants a
"non-exclusive, worldwide, non-transferable and royalty-free limited license ... to use,
reproduce, distribute, copy, create derivative works of, and make modifications to the DINO
Materials". There is **no non-commercial clause**, no Llama-style MAU trigger, and no field-of-use
restriction beyond the acceptable-use items (§1.b.v: no ITAR/military/warfare, nuclear, espionage,
guns/illegal weapons; §1.b.iii trade-control compliance). §5.a is helpful for Rule 3.14.a: "with
respect to any derivative works and modifications ... you are and will be the owner of such
derivative works and modifications."

Three concrete obligations that follow, and one is **already breached in the staged payload**:

1. **§1.b.i redistribution**: "If you distribute or make the DINO Materials, or any derivative
   works thereof, available to a third party, you may only do so under the terms of this
   Agreement and you shall provide a copy of this Agreement with any such DINO Materials."
   → `kaggle/slotknee_code/weights/` currently ships `vit_small_patch16_dinov3.lvd1689m.safetensors`
   with **no licence file anywhere in the directory** (`find` returns nothing). Before any push
   that includes it — private dataset or public — a copy of `LICENSE.md` must sit beside it. Same
   applies to the **fine-tuned** checkpoints if DINOv3 ever backs a final entry: those are
   derivative works and must be published under the DINOv3 Agreement, not CC-BY-NC 4.0.
2. **Rule 2.5's carve-out covers exactly this**: an incompatibly-licensed pretrained model does
   not have to be relicensed. So a DINOv3-backed winning entry is coherent — our code goes
   CC-BY-NC 4.0, the DINOv3-derived weights go out under the DINOv3 Agreement, and 2.8's "public
   kaggle dataset" obligation is still satisfiable because the DINOv3 Agreement *permits* public
   redistribution (unlike a no-derivatives or no-redistribution dataset licence).
3. **§1.b.ii** requires acknowledging DINOv3 in any publication of results — trivially satisfied
   by the winners' writeup, but put it there deliberately. **§8** lets Meta modify the Agreement
   unilaterally with immediate effect; re-read it before the Nov 5 winners' deadline if DINOv3 is
   in a final entry.

**So POLICY 4's live example resolves in our favour**: DINOv3 may enter a final submission. What
it needs is not a host ruling but a **packaging fix** (ship the licence) and a line in the writeup.

### 5. The two licence traps that are *not* about permission to train

Both concern Rule 2.8 (publish weights publicly) and Rule 3.14.a ("unrestricted right to grant"),
which 2.6(a) explicitly says permission to use external data does **not** waive:

- **ND (NoDerivatives)** — kneeMRI/Štajduhar is CC BY-NC-**ND** 4.0. Whether a model trained on a
  dataset is a "derivative work" of it is genuinely unsettled and jurisdiction-dependent. Training
  is defensible; **publishing the resulting weights publicly, as 2.8 requires, is where ND bites.**
- **SA (ShareAlike)** — RadImageNet is CC-BY-NC-**SA**-4.0. Share-alike would try to force our
  published weights under CC-BY-NC-SA, which collides with the CC-BY-NC 4.0 grant in 2.5 and with
  warranting an "unrestricted right to grant". Note 2.5's carve-out says we need not grant an open
  licence for such a model — it does **not** say the model's own licence stops applying to us.

NC alone is comparatively harmless **in this competition specifically**, because the winner licence
is itself CC-BY-NC 4.0. A participant raised the opposite worry (that prize money makes it
commercial use); that reading is the one that needs a host answer, not the licence-compatibility one.

### 6. The lever that is unambiguously free

**Self-supervised pretraining on the competition's own 4,407 studies needs no external data at
all** and touches none of the above. The competition data is licensed to us for exactly this
(§1.7 "Commercial and Academic Research - MIRA license"; 2.4.a "You may access and use the
Competition Data for any purpose"), the resulting encoder is ours to publish under 2.8 with no
third-party licence riding on it, and the `slotknee-ssl` session is already scheduled (W1/B3, ~2 h,
gate D5). Our 6-slot × 9-slice cache alone is ~238k slice-crops from those 4,407 studies before any
external corpus is considered. **If external SSL data is wanted on top, KMAR-50K is the only
candidate that is clean today.** Do not spend a session waiting on a host ruling that has gone
unanswered four times.

### 7. What needs Felipe's decision

1. **Ask the Host, or don't.** The narrow question — "has the Host specifically prohibited any
   public knee-MRI corpus under 2.6.b, and can weights derived from an NC/ND-licensed corpus
   satisfy 2.8(a)'s public-weights obligation?" — is the only thing that unblocks MRNet / OAI /
   SKM-TEA / fastMRI. Four participants have asked variants and got silence for 8+ days. Felipe's
   call: post a fifth, or write the gated corpora off for this competition. **Recommendation:
   write them off** — the campaign has no schedule slack that depends on them, and §6 makes them
   unnecessary.
2. **KMAR-50K (CC BY 4.0, ungated, ~10 GB): fetch or not?** This is the only external SSL corpus
   that needs no ruling. Blocked today only by the deploy freeze and disk (11 GiB free vs a 9.73 GB
   part 1). Needs an explicit authorisation because it is new external data under POLICY 2's
   logging requirement.
3. **kneeMRI ND clause.** Usable for measurement now. A judgement call is needed only if it would
   feed a prize-eligible run, because of 2.8's publish-the-weights obligation.
4. **DINOv3 packaging.** POLICY 4 can be relaxed to "DINOv3 is cleared for final submissions,
   conditional on shipping `LICENSE.md` with the weights and acknowledging it in the writeup".
   Someone must make that packaging change — this scout did not touch `kaggle/`.
5. **RadImageNet is a trap, not an opportunity** (`docs/gold_medal_plan.md` lists it as an optional
   upgrade). CC-BY-NC-SA share-alike + unanswered #735121. Recommend closing it as a candidate.
6. **Caveat on the rules text itself**: Specific Terms §1.4 gives the Competition Website as
   `.../rsna-knee-abnormalities-detection` (plural) and §1.1 the title as "RSNA Knee Abnormalities
   Detection", neither of which matches the live slug `rsna-knee-abnormality-detection`. Almost
   certainly a host typo; noted only so nobody thinks they are reading a different competition's rules.

## Where we actually stand vs the published state of the art (main, 2026-08-24 19:00)

**CoPAS** (Qiu et al., *Nature Communications* 2024, HKUST + Southern Medical University) is the
closest published work to this competition: **the same task, twelve knee abnormalities**, a
purpose-built architecture, 1,748 patients across five centres, arthroscopy-confirmed labels.

- CoPAS overall **AUC 0.812**, per-finding 0.734–0.877.
- It beats junior radiologists (0.65 accuracy) and matches seniors (0.80).
- **Our champion: 0.8531 pooled OOF / 0.866 public LB.**

Different datasets, so not a like-for-like comparison — but it recalibrates "are we any good?".
We are not behind the research frontier on this task; a 0.95 leaderboard is far more likely to
reflect cleaner competition labels and heavy ensembling than a modelling gap we have failed to
close. It also means the levers below are refinements to an already-strong model, not rescue work.

**Two findings from CoPAS worth acting on:**
1. **MCL is a KNOWN hard case, not our private failure.** CoPAS reports MCL at 0.782, among its
   worst, and diagnoses why: "certain classes (ACL, MCL, CONT...) exhibit opposite trends because
   the model yields similar results in three planes" — i.e. the model fails to become
   plane-specific for these findings. That is precisely the mechanism the coronal joint-line zoom
   attacks: force a dedicated coronal view instead of hoping attention discovers plane specificity.
2. **Their architecture's advantages, we largely already have.** Their "plane-aware probability
   matrix" (per-finding x per-plane fusion) is what our per-finding attention over (slot, anchor)
   tokens already computes, and their multi-task framing is our 12-label joint head. The genuinely
   novel pieces are cross-plane spatial attention and cross-sequence conditioning (using T1W as a
   contrast reference for the fluid-sensitive sequence); our mixer layer is a general form of both,
   which is some evidence the mixer earns its keep (`nomixer` remains an ungated arm — worth running).

## Label re-extraction feasibility (scout, 2026-08-24)

**Question.** The public LLM labels (`data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv`)
score macro **0.9089 AUC** on the 58 gold studies over the 10 clean findings (0.8927 over all 12),
and are visibly uneven per finding. Published work says careful chain-of-thought extraction beats
weaker labelers badly on exactly the findings we are weakest on. Is re-extracting all 4,407 reports
worth doing?

**Method.** I read all 58 gold reports myself (Spanish, English, Turkish, Greek, Bulgarian, Croatian,
Dutch, German — mean 1,305 chars) and hand-authored 12 probabilities per study using explicit
reasoning rules, without looking at the gold labels first. Output:
`data_subset/labels_external/llm_reextract_gold58.csv` (58 x 12 probabilities). Scored with
`sklearn.metrics.roc_auc_score` per label, exactly as the scripts do.

### Head-to-head on the 58 gold studies

| finding | v4_blend | this re-extraction | delta | bootstrap 95% CI on delta |
| --- | --- | --- | --- | --- |
| ACL | 0.9871 | **0.9902** | +0.0031 | [-0.004, +0.014] |
| MCL | 0.9683 | **0.9830** | +0.0147 | [-0.012, +0.056] |
| Medial Meniscus | **0.9483** | 0.9417 | -0.0066 | [-0.052, +0.029] |
| Lateral Meniscus | 0.8789 | **0.8932** | +0.0143 | [-0.014, +0.052] |
| Medial OA | 0.9318 | **0.9519** | +0.0202 | [-0.019, +0.068] |
| Lateral OA * | 0.8327 | **0.8549** | +0.0222 | [-0.028, +0.075] |
| PF OA | **0.9015** | 0.8938 | -0.0077 | [-0.049, +0.026] |
| Effusion | **0.8770** | 0.8640 | -0.0130 | [-0.063, +0.032] |
| Synovitis * | **0.7903** | 0.7581 | -0.0323 | [-0.103, +0.035] |
| Baker's | **0.9438** | 0.8976 | -0.0462 | [-0.156, +0.013] |
| Contusion | 0.8596 | **0.8603** | +0.0007 | [-0.054, +0.052] |
| **Fracture** | 0.7931 | **0.8944** | **+0.1014** | **[+0.008, +0.215]** |
| **MACRO12** | 0.8927 | **0.8986** | +0.0059 | [-0.010, +0.021] |
| **MACRO10** (excl. Lateral OA, Synovitis) | 0.9089 | **0.9170** | +0.0081 | [-0.009, +0.025] |

`*` = findings with verified gold errors; reported, never tuned against.

Variants: `v4_blend` with only the Fracture column replaced → MACRO12 **0.9012** / MACRO10 **0.9191**
(the best of anything tried). A 50/50 average of the two label sets → 0.8960 / 0.9133, i.e. worse
than the Fracture-only swap — the two sets are not complementary.
Note also that `llm_labels_v5_repaired.csv` differs from v4_blend on the full 4,407 but is
**bit-identical on all 58 gold studies**, so it is untested by any gold measurement.

### The decisive result: the ceiling is the reports, not the extractor

`v4_blend` predicts **my** binarised labels at macro AUC **0.999** (1.000 on nine of twelve
findings) — two fully independent extractors reading the same text agree with each other almost
perfectly. Yet both score only ~0.89-0.90 against gold.

Sharper still: I confidently contradict gold (my p >= 0.85 where gold = 0, or p <= 0.10 where
gold = 1) in **76 of 696 cells (10.9%)**, and `v4_blend` sides with **me against gold in 74 of
those 76 (97%)**. Spot checks confirm these are not extraction misses:

- idx 56 — report is explicitly all-normal ("No significant knee effusion... Normal visualized
  bones and surrounding soft tissues"). Gold `Baker's = 1`.
- idx 50 — "Medial meniscus tear... Cartilages normal... No Baker's cyst. Mild effusion."
  Gold `Medial Meniscus = 0`, `Lateral OA = 1`, `Synovitis = 1`, `Contusion = 1`.
- idx 11 — "Rotura de menisco interno y lateral... Contusiones oseas femorotibiales."
  Gold `Medial Meniscus = 0`, `Contusion = 0`, `Fracture = 1`.
- idx 48 — German report names a Baker cyst with measurements; gold `Baker's = 0`.

So the ~0.09 AUC gap between the labels and gold is **not extraction error**. Gold is not a
function of the report text — it is either image-derived / independently re-read, or noisier than
the two flagged findings suggest. The known-bad-label caveat is real but **understates** the
problem: it is not confined to Lateral OA and Synovitis. The Baker's "loss" (-0.046) is entirely
gold errors that v4 happens to rank luckily, not a worse read.

**Corollary: 0.909 is not a label-quality score.** It is the agreement ceiling between report text
and gold. A perfect report reader cannot go much above it.

### Reasoning rules that measurably worked (portable to any re-extraction prompt)

1. **Fracture is the one real win, and the rule is granularity, not recall.** v4 dumps every
   unmentioned/uncertain case into a single 0.25 plateau; AUC ties there cost it heavily. Grading
   `subchondral insufficiency fracture` / `impaction fracture` / `bony avulsion` / `kirik` /
   `fraktura` high (0.85-0.98) but `osteochondral impaction injury` **intermediate** (0.45-0.70)
   when the report also says "no acute fracture" broke five ties in my favour. +0.101 AUC, the
   only per-label delta whose 95% CI excludes zero.
2. **Intrasubstance signal is not a tear.** "aumento de senal ... que no impresiona contactar con
   la superficie articular", "without extension to the articular surface", "grade I-II
   degeneration" -> 0.12-0.25, not 0. Both extractors already do this; it is table stakes.
3. **"Amputacion" / "truncated" / post-meniscectomy with recurrent tear IS abnormal** (0.85-0.96).
4. **Compartment discipline.** "lateral trochlea" and "medial patellar facet" are *patellofemoral*,
   not lateral/medial tibiofemoral. Conflating them is the main avoidable source of OA error.
5. **Contusion vs fracture vs degenerative marrow oedema.** Oedema in a trauma report -> contusion
   0.85-0.96; oedema adjacent to an insufficiency fracture or "related to overlying chondrosis" ->
   0.25-0.45; "no bone bruise" -> 0.05.
6. **Report contradictions are information, not noise.** idx 26 says "No evidence of knee effusion"
   in findings and "Mild effusion" in the conclusion -> 0.55, not 0 or 1. Soft targets earn their
   keep here.

### Per-label verdict

- **Adopt:** Fracture only. Statistically significant, mechanistically explained, and the
  Fracture-only swap is the single best label set measured (MACRO10 0.9191, +0.010 over baseline).
- **No evidence either way:** ACL, MCL, Lateral Meniscus, Medial OA, Contusion — deltas positive
  but every CI spans zero on n=58.
- **Do not adopt:** Medial Meniscus, PF OA, Effusion, Baker's — I lose, and inspection shows the
  losses are gold errors ranked luckily, not better reading. Chasing them is tuning to noise.
- **Unmeasurable:** Lateral OA, Synovitis — gold is known-corrupt for these. Report, never tune.

### Cost and value of re-extracting all 4,407 reports

Cost is genuinely small. 4,407 reports, 4.84 M characters, ~1.5 M input tokens of report text
(char/3.2 estimate for mixed-script text; no API key on this machine, so not verified with
`count_tokens`). With a ~1,500-token cached rule prompt and ~1,000 output tokens per report
including reasoning, one full pass on a frontier LLM via a batch API (50% off, $2.50/$12.50 per
MTok) is roughly **$55-70**; three-sample self-consistency roughly **$150-200**; on
a smaller model, about a third of that. Wall-clock is a few hours of batch turnaround plus
maybe half a day of prompt/harness work. **Cost is not the barrier.**

Value is the problem. The expected return, measured:

- Macro gain from a full careful re-extraction: **+0.006 (12-label) / +0.008 (10-label)**, with a
  bootstrap CI spanning zero. That is not a detectable improvement on n=58, and it is upstream of a
  model that then has to learn from it — the downstream OOF effect would be smaller still.
- Eleven of twelve findings show no reliable gain. The one that does is worth **+0.101 AUC on
  Fracture / +0.010 macro**, and it can be captured by re-extracting **one column**, not twelve.
- The residual gap is un-attackable from text. Two independent extractors agreeing at 0.999 while
  both sit at 0.89 against gold is as clean a ceiling measurement as this dataset permits.

**Recommendation.** Do **not** re-extract all 12 findings over 4,407 reports — this is a negative
result and should be recorded as one. **Do** re-extract the single **Fracture** column over all
4,407 reports with the granularity rules in (1) above; that is a ~$10-20 job with the only
statistically defensible payoff on offer, and it lands on a finding where the model already beats
its target. Separately, `docs/known-defects.md` should be updated: the gold-58 label-error note
currently names only Lateral OA and Synovitis, but the confident-contradiction rate is ~11% of all
cells and spans Baker's, Effusion, Medial Meniscus and Contusion as well. Any future decision tuned
against gold-58 at a resolution finer than ~0.02 macro AUC is tuning to label noise.

**Reproduce:** `data_subset/labels_external/llm_reextract_gold58.csv` holds the extraction;
scoring is `roc_auc_score` per label against the non-null rows of `data_subset/train_gold.csv`.

## ⭐ THE DIAGNOSIS (main, 2026-08-24 20:40): the per-finding attention has collapsed to mean pooling

Ran `SlotKneeS.attention_report` on a trained checkpoint. The head that is the whole point of
this architecture — a learned query per finding, attending over the (slot, anchor) tokens — **is
not doing anything**:

- **Normalised attention entropy 0.9994** (1.0 = perfectly uniform = mean pooling).
- **Learned `attn_tau` 0.955–1.000**, from a 1.0 init: the temperature never sharpened.
- **Anchor attention spread (max−min over G) = 0.0085**: all anchors weighted alike.
- Per-sequence weights sit at ~1/6 each, and several findings have *identical* attention rows
  (ACL == MCL to three decimals; the two menisci identical; Effusion/Synovitis/Baker's identical).
- Where it does tilt, it tilts WRONG: ACL's largest weight is on AX_FS (ACL is a sagittal
  finding); MCL's largest is also AX_FS (MCL is coronal); both menisci favour SAG_T1.

**This single fact explains the whole campaign's results:**
- Why MCL 0.778 and Lateral Meniscus 0.748 lag — a finding visible on one plane is averaged
  across six. This is exactly the mechanism CoPAS names for its own MCL failure ("the model
  yields similar results in three planes").
- Why every capacity arm was NULL (ViT-B x2, tb12, ep14, lrb1e4): more capacity cannot help when
  the aggregation throws the spatial evidence away.
- Why COVERAGE was the biggest win (+0.0325): if the model is averaging, more samples make a
  better average. We have been improving the mean, not the selection.

**The fix was designed and never connected.** `SlotKneeS` accepts `attn_entropy_weight` — a
penalty on uniform attention, with `forward` storing a differentiable `last_attention_entropy`
for the trainer to add — but `scripts/train_slotknee.py` never exposed it, so the guard against
this exact failure has been dormant the whole time. Now wired end to end as
`--attn-entropy-weight` (default 0.0, so nothing already measured changes), smoke-tested, and
recorded in the checkpoint hparams.

**Priority order for the fix, per the model's own docstring** ("entropy is stationary at
exactly-uniform attention, so its gradient vanishes there — the temperature is the stronger
lever"):
1. `--attn-tau-init 3.0` — already queued as `r4tau3`. Strongest lever; escapes the collapse.
2. `--attn-tau-init 3.0 --attn-entropy-weight 0.05` — sharpen the init AND penalise re-collapse.
   The combination is the real candidate; entropy alone cannot escape a uniform state.
3. `--mil-pool lse` — already queued as `r4mil`; sidesteps the attention entirely for
   slice-level findings.
The queue already leads with (1) and (3), which is lucky rather than clever — they were queued
to attack "dilution" before the mechanism was confirmed. Add (2) at the next driver boundary.

## ⭐ REFRAME (main, 2026-08-24 22:15): the model is an AVERAGER, and averaging is correct for it

Three independent measurements now point the same way, and they overturn the "dilution"
hypothesis that motivated the whole attention-fix family:

1. **`r4mil` REGRESSES −0.0182** (gate ±0.0066 at the new reg4 noise floor of 0.0033) — and it
   is worst on exactly the findings it was built to rescue: ACL −0.036, MCL −0.019, LatMen
   −0.015. MIL log-sum-exp pooling is the textbook fix for "the finding is on 1–2 of 60 slices",
   and it made those three findings *worse*.
2. **Sharpening a trained model at inference degrades it monotonically**: macro 0.8089 → 0.8087
   → 0.8056 → 0.7817 at tau ×1/×3/×10/×30.
3. **The learnable temperature drifted DOWN** (1.0 → 0.955–1.000). Given the freedom to sharpen,
   the optimiser chose to flatten.

**The honest conclusion: uniform attention is not a pathology here, it is the solution.** With
4,407 studies and noisy labels, each slice's evidence is weak; averaging is variance reduction,
and a peaked read-out amplifies noise. The `scripts/model_health.py` check confirms nothing
upstream is broken — slot embeddings distinguishable (cosine −0.046), anchor embeddings
distinguishable, the mixer actively mixing (relative token change 1.02), sequences distinct
(0.317). The pipeline is healthy right up to a read-out that averages *on purpose*.

**This makes the whole campaign coherent for the first time:**
- Coverage was the biggest win (+0.0325, 18 → 60 images) because **more samples make a better
  average** — exactly what a variance-reducing averager wants.
- Every capacity arm was NULL because capacity was never the constraint.
- MIL, and probably the tau/cosine arms behind it, fail because they fight the mechanism that
  is actually working.

**Consequences for the plan:**
- **Downgrade the attention-fix family.** Let `r4tau3` and the cosine arms run — they are cheap
  and the hypothesis deserves a clean test rather than an inference-time proxy — but expect
  NULL/REGRESS, and do NOT spend a T4 session on this family without a laptop ADOPT first.
- **Upgrade the input levers.** If the model improves by averaging better evidence, then MORE
  and BETTER-TARGETED images is the lever: coronal joint-line zoom (`zoomc`, targeted), G20
  (more anchors, caches already built), 336 px (more detail per image). The efficiency tension
  documented earlier is therefore the REAL constraint on this campaign, not a side note.
- The `r4sharp*`/`r4cos*` arms keep their value as a decisive, cheap closure of the question.
  Judge by entropy first: an arm that never escaped says nothing about the hypothesis.

## ⭐ PREDICTION (main, 2026-08-24 23:00): G20 is saturated — cancel it, keep zoomc and 336

Having concluded the model is a variance-reducing averager, the obvious question is how much
MORE evidence is worth. That is answerable WITHOUT a GPU session: take the champion (trained at
G=10) and sub-sample its anchors at inference.

Champion model, 150 held-out studies, 2 random subsets per G (subset spread < 0.002):

| anchors | imgs/study | macro AUC | marginal |
|---|---|---|---|
| 6 | 36 | 0.8354 | — |
| 8 | 48 | 0.8458 | **+0.0104** |
| 10 | 60 | 0.8474 | **+0.0016** |

**The marginal value of anchors has collapsed by 6.5x between 6→8 and 8→10.** We are on the flat
part of the curve. Extrapolating, 10 → 20 anchors buys roughly +0.002–0.004 — below the 0.0066
gate — while costing 2x the inference time (−0.021 AUC-equivalents on the efficiency score).

**Decision: do NOT spend a T4 session on the G20 all-slices gate.** The `g20a`/`g20b` caches are
already built and can stay on Kaggle at no cost, but the arm should drop to the bottom of the
queue. Caveat recorded honestly: a model TRAINED at G=20 might learn to use the extra anchors
differently than this trained-at-G=10 probe suggests — but with saturation this sharp, the prior
is strongly against, and one T4 session is too expensive for that prior.

**This SHARPENS the input-lever conclusion rather than reversing it.** "More images" is not the
lever — it was, between 18 and 48 images, which is exactly the range where coverage won
(+0.0325). That well is now dry. What remains untested is a different KIND of evidence:
- **Better-targeted images** — the coronal joint-line zoom (`zoomc`, cache building): new
  information about the plane where MCL and the menisci live, not more samples of the same view.
- **More detail per image** — 336 px at full coverage: a different axis entirely, and the one
  the radiology-transfer literature says matters for small structures.
Both remain live. G20 does not.

Method note worth keeping: sub-sampling a trained model's inputs at inference is a cheap way to
price a data-scaling experiment BEFORE paying for it. It cost ~10 minutes of CPU and saved a
GPU session. Consider it for the 336 question too (the inverse — downsample the images and
watch the slope), though resolution cannot be probed as cleanly as anchor count.

## ⭐ THREE LEVERS PRICED AND CLOSED WITHOUT A GPU SESSION (main, 2026-08-24 23:45)

Same method throughout: interrogate the TRAINED champion to predict what an experiment would
buy, before paying for it.

**1. More anchors (G20) — SATURATED.** Sub-sampling the champion's anchors at inference:
6 → 0.8354, 8 → 0.8458 (+0.0104), 10 → 0.8474 (+0.0016). Marginal value collapsed 6.5x. G20 is
predicted at +0.002–0.004, under the 0.0066 gate, for 2x the inference cost. **Closed.**

**2. More resolution (336 px) — the model does not use the detail it already has.** Gaussian
low-pass on the native 224 px inputs (no resampling artefacts, so every point is comparable):

| blur sigma | ~detail kept | macro AUC | cost |
|---|---|---|---|
| 0.0 | native 224 px | 0.8387 | — |
| 0.5 | ~112 px | 0.8375 | **−0.0012** |
| 1.0 | ~75 px | 0.8269 | −0.0118 |
| 2.0 | ~45 px | 0.8083 | −0.0304 |

**Erasing everything finer than ~112 px costs 0.0012.** The model's evidence lives at coarse
scale; it is not exploiting the resolution it already has, so giving it 2.25x more is giving it
more of what it demonstrably ignores. Consistent with the earlier p336@G4 NULL. **Closed** —
with the honest caveat that a model TRAINED at 336 might learn to use finer detail, which this
probe cannot rule out; it prices the prior, and the prior is now poor.

**3. More seeds — capped by correlation.** Two seeds of the identical recipe on identical folds:
0.8196 and 0.8163, rank-averaged **0.8207** = **+0.0011** over the better seed, with a per-label
rank correlation of **0.953** between them. Seeds see the same things. **Closed** as a lever
worth a session (still free to keep both checkpoints in an ensemble).

**What survives, and why.** Every closed lever was closed because the CURRENT model's behaviour
already determines its value. The one input lever that cannot be priced this way is the coronal
joint-line zoom (`zoomc`) — precisely because it supplies information the current model has
never seen, so no probe of that model can predict it. **The unpriced lever is the one worth
paying for.** The campaign's remaining shortlist is therefore short and honest:
1. **reg4 5-fold** — already measured at +0.0220, needs one T4 session. Highest certainty.
2. **zoomc** — the only genuinely untested KIND of input.
3. The training-time levers running tonight (distillation, AUCM, SAM), which cost 0 s at
   inference and so cannot hurt the efficiency entry.

## r4distill = ADOPT +0.0185 — with a leak caveat I have to raise against my own result

`r4distill` (reg4 backbone + `--distill-targets teacher_g10_oof.csv --distill-weight 0.5`):
**0.8381 vs 0.8196 = +0.0185**, gate +0.0066, uid-ovl 1.0000. Every label up; Baker's +0.089,
Contusion +0.020, Fracture +0.022, MedMen +0.020, ACL +0.016, MCL +0.013. On raw size this is
the second-largest win of the campaign after coverage, comparable to reg4 — and unlike the
input levers it costs **0 s at inference**, so it is free on the efficiency track too.

**But the protocol is not as clean as I claimed when I built the teacher.** I called
`teacher_g10_oof.csv` "leak-free" because each study's target comes from the g10 fold-model that
did NOT train on that study. That removes *memorisation*, but not *cross-fold information flow*:

- Validating on fold 0, the student trains on folds 1–4.
- The teacher target for a fold-1 study came from the g10 model trained on folds 0,2,3,4 —
  a model that **saw fold 0 and its labels**.
- So fold-0 label information is embedded in the teacher's learned function, passed to the
  student through the targets, and the student is then scored on fold 0. **Indirect leak.**

A properly clean test needs, for validation fold k and training study in fold j, a teacher that
saw neither k nor j — i.e. leave-TWO-folds-out teachers, which we do not have and which cost a
full extra training round to produce.

**Status: PROMISING, NOT BANKED.** +0.0185 is an upper bound; the true effect is somewhere
between that and zero. Do not put distillation into a final recipe on the strength of this arm
alone. It is exactly the failure mode that got the ORIGINAL distill arm (+0.0127) excluded, and
I reproduced a weaker form of it while believing I had fixed it.

**How to settle it, cheapest first:**
1. **Public LB.** A 5-fold distilled run submitted once answers it directly — the hidden test
   set shares no folds with anything. One GPU session + one submission, and it is the same
   session that would produce the entry anyway.
2. Leave-2-folds-out teachers on the laptop (expensive, ~2x the arm cost, fully local).
3. Note the shape of the evidence: the gain is spread across ALL twelve labels rather than
   concentrated, which is more consistent with a real regularisation effect than with a leak
   (a leak would favour whichever labels the teacher fits best). Suggestive, not decisive.

## Do the three wins STACK? (main, 2026-08-25 00:30) — mostly not, and that reprices the next session

Correlating the per-label gains of the three measured wins tells us whether they fix different
things (additive) or the same thing (sub-additive):

|  | coverage | reg4 | distill |
|---|---|---|---|
| mean gain | +0.0341 | +0.0220 | +0.0185 |
| corr with coverage | — | **+0.458** | **+0.829** |
| corr with reg4 | +0.458 | — | +0.359 |

**Distillation and coverage are 0.829 correlated — they fix nearly the same thing.** The
fingerprint is unmistakable: Baker's is coverage's biggest win (+0.094) and distillation's
biggest win (+0.089); the labels coverage helps most are the labels distillation helps most.
Both are VARIANCE-REDUCTION mechanisms — coverage averages more images, distillation smooths
the targets — which is exactly consistent with the "this model is an averager" reframe.

**Consequence:** distillation's +0.0185 was measured on the BASE layout (G=3), which lacks
coverage. On the champion, which already HAS coverage, most of that benefit is already banked.
Expect roughly 30-50% to survive: **+0.006 to +0.009**, not +0.0185.

**Revised forecast for the next 5-fold session (coverage + reg4 + distillation):**

| component | naive | overlap-discounted |
|---|---|---|
| champion (coverage) | 0.8531 | 0.8531 |
| + reg4 (r=0.458 with coverage) | +0.0220 | +0.013 … +0.016 |
| + distillation (r=0.829 with coverage) | +0.0185 | +0.006 … +0.009 |
| **pooled OOF** | 0.894 | **0.872 … 0.878** |
| **public LB** (+0.018 ensemble gap) | ~0.91 | **0.890 … 0.896** |

So: expect **~0.89**, not ~0.91. Still the largest single step available, and worth the session.

**Two further consequences worth acting on:**
1. **Mild evidence the distillation result is NOT a leak artefact.** Its per-label fingerprint
   matches coverage's — a legitimate variance-reduction pattern. A leak would favour whichever
   labels the teacher fits best, not reproduce the shape of an unrelated, physically-motivated
   intervention. Does not remove the need for the LB check, but it raises the prior.
2. **The remaining headroom is in mechanisms that DON'T correlate with coverage.** Everything
   that reduces variance is now largely spent (coverage itself, distillation, seeds, anchors —
   all saturating or overlapping). `zoomc` is the one candidate that adds *new information*
   rather than averaging existing evidence better, which is precisely why it is uncorrelated
   with anything already measured — and why it is now the most valuable untested thing we have.

## Ensemble selection implemented — and its first verdict is "do not ensemble" (2026-08-25 01:15)

`scripts/select_ensemble.py`: greedy forward selection WITH REPLACEMENT (Caruana et al.) over
any set of OOF runs, scored honestly (select on 4 folds, evaluate on the held-out fold, rotated).
Built because with capacity, resolution, anchors and seeds all measured out, a diverse stacked
ensemble is the largest remaining lever — and because "throw every checkpoint in" is measurably
wrong here: the 10-member two-view entry scored 0.864 against 0.866 for a clean 2-member one.

**First run, over the five reg4-family runs we have:**

| run | solo macro |
|---|---|
| r4distill_s42 | **0.8381** |
| reg4_s42 | 0.8196 |
| r4tau3_s42 | 0.8186 |
| reg4_s1337 | 0.8163 |
| r4mil_s42 | 0.8020 |

Greedy picked **r4distill 5 times out of 5** — it refuses to mix anything in — and the held-out
gain over the best single model is **+0.0000**.

**That is the finding, not a failure of the tool.** These five runs are the same architecture on
the same layout with small knob changes; we already measured seed-to-seed rank correlation at
**0.953**. Ensembling near-identical models cannot add information, and the selector correctly
declines rather than diluting. (This is exactly the behaviour that would have prevented the
10-member entry from scoring BELOW the 2-member one.)

**Consequence for the campaign plan.** The +0.015–0.025 attributed to "diverse ensembling" is
real only if the members are genuinely diverse. Seeds, temperatures and pooling variants of one
recipe do NOT qualify. What would:
- a different LAYOUT (`zoomc` coronal zoom — cache built and waiting),
- a different BACKBONE (`dinov3s` — implemented and staged, ~23% faster too),
- a different RESOLUTION or slice structure (the untested `g10t3` local-3D axis).
Build those in the middle weeks, then run this selector: it will say honestly whether they add
anything, BEFORE a session is spent on a final ensemble that dilutes.

## TTA priced (main, 2026-08-25 02:30) — keep it for accuracy, DROP it for efficiency

Nobody had ever measured what the 3-shift anchor TTA buys, though it costs 2 extra forward
passes per model on every submission. Measured directly: champion fold-0 model, 131 held-out
studies rebuilt from DICOM at five anchor shifts.

| TTA | passes/model | macro AUC | marginal |
|---|---|---|---|
| shift 0 only | 1 | 0.8328 | — |
| shifts 0,+1,−1 (**what we ship**) | 3 | 0.8372 | **+0.0044** |
| shifts 0,±1,±2 | 5 | 0.8378 | +0.0006 |

**TTA is real but saturated at 3 shifts** — going to 5 adds +0.0006 for another 2 passes. Do not
add more shifts. (n=131, so ±0.01 noise on any single number; the SHAPE — a real first step then
a flat one — is the trustworthy part.)

**The actionable half is the efficiency arithmetic.** At 0.01 AUC ≈ 717 s:

| entry | model time | AUC | runtime cost | NET |
|---|---|---|---|---|
| 1 shift | 0.392 s/study | — | — | — |
| 3 shifts | 1.176 s/study | +0.0044 | **−0.0142** | **−0.0098** |
| 5 shifts | 1.960 s/study | +0.0050 | −0.0284 | −0.0234 |

**TTA is NET NEGATIVE on the efficiency score.** It buys 0.0044 of AUC and spends 0.0142 of
runtime. So the two entries genuinely diverge:
- **ACCURACY entry**: keep 3-shift TTA. Runtime is irrelevant under the 9 h cap, so +0.0044 is free.
- **EFFICIENCY entry**: run with `SK_TTA=0`. The submit kernel already reads that env var, so this
  is a setting, not a code change — and it is worth ~+0.010 on that track.

Method note: this is the third lever priced by interrogating the trained champion rather than
training something (after anchor saturation and detail sensitivity). All three changed a
decision; none needed a GPU session.

## ⭐⭐ THE DEEPEST FINDING (main, 2026-08-25 04:00): the labels are excellent; the lesion signal is destroyed at the encoder output

Two things were measured tonight that, together, relocate the entire bottleneck.

**1. The labels are NOT the bottleneck — they are near-perfect.** Two INDEPENDENT full
extractions (stevenleehans v4_blend and pilkwang report_labels_v2) agree with each other at
**0.979 mean AUC over all 4,406 studies**; the worst finding is Contusion at 0.927 and eight of
twelve are above 0.98. Our model scores **0.831** against those labels. This overturns the
project's founding belief ("labels are the RSNA bottleneck") — extraction is reliable, and there
is ~0.15 of headroom that belongs to the MODEL. Caveat kept honest: inter-extractor agreement
measures report→label fidelity, not image→label achievability; the true model ceiling is lower.
But the public leader at 0.952 proves ~0.95 is reachable from images, so the headroom is real.

**2. The model is not overfitting, and barely undertrained.** Every run's val AUC is still
rising and val loss still falling at the last epoch — no divergence anywhere. But ep14 shows the
curve flattening: epochs 1→8 bought +0.061, epochs 8→14 only +0.0063. Longer training is worth
perhaps +0.01, not the missing 0.15.

**So where does 0.15 live?  In a dilution nobody had looked at: WITHIN the image.**

`_pool_tokens` squashes each image's **256 patch tokens into CLS + MEAN**. A meniscus tear or an
MCL sprain occupies perhaps **3-8 patches**, so its evidence is attenuated **~30-80x at the
encoder output** — before slots, before the mixer, before attention, before any MIL machinery.

This single fact explains the pattern no other hypothesis fit:
- **Why MIL pooling REGRESSED (−0.0176)**: it attacked the SECOND dilution (across 60 tokens)
  while the FIRST had already destroyed the signal. Sharpening a read-out over vectors that no
  longer contain the lesion cannot help, and costs capacity.
- **Why resolution is ignored** (blur below ~112 px costs 0.0012): finer detail lands in more
  patches, all of which get averaged away regardless.
- **Why more anchors saturate**: more images, same per-image destruction.
- **Why the weak findings are the small ones** (LatMen 0.748, MCL 0.778) while large, diffuse
  findings do well (Effusion 0.867, Baker's 0.914) — big findings survive averaging; focal ones
  do not.

**Implemented**: `pool="cls_mean_max"` (concat CLS, mean, and **max** over patches) and
`cls_mean_topk` (mean of the k most-activated patches — a softer, less noise-sensitive peak),
exposed as `--pool` / `--pool-topk`. Cost: **+0.1M head parameters and ZERO extra encoder time**,
so it is free on the efficiency track. Default unchanged (`cls_mean`), hparams record it, all
five modes verified to build, run and round-trip.

**Queued first** in `scripts/queue_diversity_arms.sh` as `r4max` and `r4topk`. This is the most
promising untested idea on the board: it is the only hypothesis that explains all four anomalies
above, it targets exactly the findings that lag, and it costs nothing at inference.

### Counter-evidence against the patch-dilution hypothesis (main, 2026-08-25 05:00)

Before the `r4max`/`r4topk` arms finish, I probed the hypothesis directly: frozen reg4 encoder,
400 studies, per-image features pooled two ways, then a cross-validated logistic probe per label.

| finding | cls_mean | + max | delta |
|---|---|---|---|
| Lateral Meniscus | 0.646 | 0.619 | **−0.028** |
| ACL | 0.702 | 0.690 | −0.011 |
| MCL | 0.650 | 0.651 | +0.001 |
| Lateral OA | 0.667 | 0.685 | +0.017 |
| **mean over 12** | **0.695** | **0.690** | **−0.005** |

**Adding the max statistic makes the frozen features slightly WORSE, and the weak findings do
not improve** — lateral meniscus, the finding the hypothesis was built to explain, is the biggest
loser. That is the opposite of the prediction.

Honest caveats that keep the arms worth running: the probe uses a FROZEN encoder while the real
model fine-tunes four blocks (fine-tuning could learn to use a peak channel a frozen encoder's
peaks do not expose); a linear probe is far weaker than the real head; and it uses the base
G3T3 cache, not the coverage layout. But the probe averages per-image features across images
exactly as the collapsed attention does, so it is a fair proxy for our model's actual behaviour —
which makes this counter-evidence, not a technicality.

**Status: the hypothesis is now UNLIKELY, downgraded from "the deepest finding".** It explained
four anomalies elegantly, which is precisely why it needed testing rather than adopting. The
`r4max`/`r4topk` arms are already running and will settle it with fine-tuning in the loop; expect
NULL. If they confirm the probe, the honest conclusion is that we still do not know where the
0.15 of model headroom lives — and saying so is better than believing a story that measurement
does not support.

## ⭐⭐⭐ COMPETITIVE INTELLIGENCE (main, 2026-08-25) — read this before any more architecture work

Read the competition's own discussion forum. It is worth more than any literature search, and it
repositions us considerably.

**Where the field actually is** (self-reported, from the "Best single-model score" thread):

| team (LB rank) | score | setup |
|---|---|---|
| Scott Willis (**4th**, also **top of the Efficiency LB**) | **0.947** 5-fold / 0.938 single fold | "started with as small a model as possible… aiming for the efficiency LB"; "spent a lot of time trying to figure out how to work around the low quality labels" |
| Tucker Arrants (23rd) | **0.942** | 5-fold **@ 224 px** |
| diet1236364 (54th) | 0.929 | 5-fold @ 224 px, LLM labels only |
| k256.dev (472nd) | 0.926 | 5-fold @ 336 px |
| Tim Krige (393rd) | 0.92 | single model, OOF ≈ LB |
| Tom Aindow (18th) | 0.915 | DINOv2, 392 px, 150 mm crop, **random bag of 32 slices per study per epoch** |
| roy214 | 0.887 | DINOv2 |
| **US** | **0.866** | 224 px, 140 mm crop, fixed 6×10 anchors |
| Berat Kirbiyik (1102nd) | 0.826 | 192 px, 9 slices, 4 slots |

**Three conclusions that change the plan:**

1. **Resolution is NOT the differentiator — 0.942 is reached at 224 px, our own resolution.**
   Our detail probe (blur below ~112 px costs 0.0012) was right, and the 336 px cache can stay
   shelved. Two independent lines of evidence now agree.

2. **Fixed anchors may be the architectural mistake.** Tom Aindow (0.915) *randomly samples a
   bag of 32 slices per study, per epoch*. We take the SAME 10 geometric anchors per slot every
   epoch. Random bagging gives (a) free augmentation, (b) exposure to EVERY slice across epochs
   rather than 60 fixed ones, and (c) an ensemble-like averaging at inference. This is the one
   input-side idea the field uses that we have never tried, and it is orthogonal to everything we
   priced out (anchor COUNT saturates; anchor SAMPLING is a different axis entirely).

3. **The labels ARE the bottleneck after all, and I had the polarity backwards.** The
   competition's targets are EXPERT ANNOTATIONS (see the Acknowledgements: ~19 skeletal
   radiologists annotated the set); the reports are a second modality, not the label source. So
   when our re-extraction "confidently contradicted gold in 10.9% of cells", that was NOT gold
   being wrong — it was the REPORT disagreeing with the expert. Report-derived labels are a
   proxy with real error, which is exactly why the 4th-place competitor says he "spent a lot of
   time working around the low quality labels" and Tim Krige warns "LLMs are very wrong a lot of
   the time here without careful guidance". **Retract the earlier "labels are excellent (0.979)"
   conclusion**: 0.979 measured extractor-vs-extractor consistency, not label-vs-truth accuracy.

**Efficiency-track reality check**: the Efficiency LB is currently led by Scott Willis at **0.947**
— not by a fast weak model. Our 0.88-at-21-min plan does not contend with that. The efficiency
prize is being fought by someone with both accuracy AND speed, exactly as the break-even table
predicted (a 0.952 entry under 107 min beats a 0.88 entry at 21 min).

**Revised priorities**: (1) random slice bagging, (2) label quality against the 58 EXPERT
annotations, (3) everything else. Both were mis-ranked by our own internal analysis; the forum
corrected both.

## Architecture research round 3 (main, 2026-08-25) — two confirmations, one new lever

**1. Our collapsed attention is VINDICATED by the literature, not indicted.** A 2026
multi-dataset MIL benchmark (arXiv 2604.26807) finds that **simple mean pooling, with no
learnable attention, matches or outperforms attention-MIL and 3D CNNs on 4 of 6 tasks while
training 25x faster**. Our head collapsed to a uniform mean and every attempt to sharpen it
failed (MIL −0.018, temperature −0.001, max-pool −0.010, top-k −0.012). That is not a defect to
fix — it is the field's strong baseline arrived at by gradient descent. **Close the attention
family permanently** and stop spending arms on it.

**2. Bag augmentation is the right instinct, independently.** The same literature notes that
generating bags of instances per patient "expanded the total training data size as well as
increased the variance in the training data" — exactly what `--anchor-bag` does, and what the
0.915 competitor described. Already implemented and training.

**3. NEW LEVER — domain-specific pretraining.** `OrthoFoundation` (arXiv 2601.18250, Jan 2026)
is a **DINOv3 backbone pre-trained on 1.2 M unlabeled knee X-ray and MRI images**, reporting SOTA
across 14 downstream tasks including **first place on MRI structural injury detection**, and
"matching supervised baselines using only 50% of labeled data". That is our exact anatomy and
close to our exact task. **Weight availability is NOT confirmed** — the paper is CC-BY-NC-ND and
no GitHub/HF release was found — so treat it as a WATCH item: if the weights appear publicly, it
is the single largest architectural upgrade on the board, because it replaces a generic
natural-image encoder with one that already knows knee anatomy.

**The accessible version of that idea is one we already built and never ran.** If a knee-specific
encoder is worth first place on structural injury detection, then self-supervised pretraining on
**our own 4,407 studies x 60 slices = ~264k knee MRI images** is the poor-man's OrthoFoundation —
and the `slotknee-ssl` kernel exists, is written, and has never been executed (it errored on the
P100 fast-fail guard and was never re-clicked). This reprices SSL from "a maybe" to the
best-justified untested lever we own, and it needs no external data, no licence, and ~2 GPU hours.

**4. Also noted, lower priority**: MM-DINOv2 (arXiv 2509.06617) adds modality-specific patch
embeddings plus full-modality masking to handle MISSING sequences, reporting +11.1% over
supervised SOTA on multi-sequence brain MRI. We have exactly that problem — AX_T1 is present in
only 19% of studies, COR_T1 in 77% — and currently handle it with a shared encoder plus slot
dropout. A per-sequence adapter is a real architectural idea, but it is a bigger change than
anything else here and should wait behind SSL and the queued arms.
