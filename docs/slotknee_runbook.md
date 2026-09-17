# SlotKnee-S runbook (how to run everything, and what each piece is for)

Scoreboard: fold-0 single **0.799** → 5-fold T4 base **0.835** → coverage (g10t1) 2-member **0.866** →
**two-band 16-epoch ensemble (central t35 + coverage g10t1, TTA) 0.905 public LB (2026-09-16)**, with the
one full-fit distilled student at **0.907** in the efficiency track the same day. Pooled OOF of the
accuracy entry is 0.8843 (0.8851 as the kernel actually combines), so the OOF→LB offset is +0.021.
Local pooled OOF on the LLM labels is the decision metric (POLICY 1: final picks by OOF) — with the
caveat the 2026-09-16 review raised: it cannot price TTA, fold-averaging or full-fit, and its gate
cannot resolve deltas below ~0.004 (see `docs/red_team_20260916.md`).

**The winning recipe** (reproducible from this repo alone): `slotknee-cache-builder-g10t1`
(P=224, G=10, T=1, 140 mm crop) → `slotknee-train-g10` (5 folds × 8 ep, bs 8, seed 42,
**`--backbone vit_small_patch14_reg4_dinov2.lvd142m`** (reg4 ADOPT 2026-08-24, +0.0187 pooled / ACL +0.053),
v4_blend labels + v2 `--weights-from`, aux 0.2, EMA 0.998, grouped folds via
`fold_image_uids.txt` symlinks) → `slotknee-submit` rank-average.

## The pieces

| piece | what it is | where |
|---|---|---|
| Slot cache | uint8 memmap `[N, slots, anchors, T, P, P]` built once from DICOM | Kaggle kernels `slotknee-cache-*` → `cache/*_full/` locally |
| Model | DINOv2-S/14 encoder (last 4 blocks trainable) + per-finding attention over (slot, anchor) tokens | `src/slotknee.py` |
| Labels | LLM-read report labels (v4 blend targets, v2 silence weights) + 58 gold ×8 | `data_subset/labels_external/` |
| Train | memmap dataset, slot/anchor dropout, EMA, aux slot loss, grouped folds, distillation | `scripts/train_slotknee.py` |
| Infer | DICOM → header-first decode → model; seeded submission first; per-layout groups; TTA | `scripts/infer_slotknee.py`, `kaggle/submit_kernel/` |
| Ablations | resumable driver, pooled-OOF table with a seed noise floor | `scripts/ablate_slotknee.sh`, `kaggle/ablate_kernel/` |
| Kaggle ops | package code dataset, push kernels, fetch outputs, submit | `kaggle/kaggle_ops.py` |

## Kaggle: the one thing the API cannot do

The competition bans P100s and Kaggle's API always assigns a P100 to a GPU kernel. Every
GPU kernel therefore fails fast (by design) when pushed from here, and must be run from the
Kaggle UI: open the kernel → **Settings → Accelerator → GPU T4 x2 → Save Version → Save & Run All**.
CPU kernels (cache builders, the CPU submit) run fine from the API.

Kernels in the account (`felipedeleon11/…`):

| kernel | purpose | how to run |
|---|---|---|
| `slotknee-cache-builder`, `-p252`, `-t30`, `-g10t1`, `-zoom` | build cache variants (CPU, ~40 min) | API: `push-kernel`, `push-cache252`, … |
| `slotknee-train` | 5 folds × 8 epochs, base recipe (~5 h T4) | UI, T4 |
| `slotknee-train-g10` | same, 60-slice coverage cache, folds 0–4 (~8 h) — **the winning recipe** | UI, T4 |
| `slotknee-train-final` | winning recipe, 10 ep, warm-start `--init-from-dir`, 2 seeds | UI, T4 |
| `slotknee-ablate`, `-g10` | 2-fold arms per session (edit `ARMS` in the kernel script) | UI, T4 |
| cache builders also: `-zoomj`, `-g20a`/`-g20b` (all-slices halves), `-p336g4` | more cache variants | API push |
| `slotknee-ssl` | MAE continued pretraining → `encoder.safetensors` (1–2 h) | UI, T4 |
| `slotknee-submit` | inference on the hidden test set; writes `submission.csv` | API (`push-submit`) or UI; T4 preferred, CPU works |

Refresh the code dataset whenever `src/` or `scripts/` change: `python3 kaggle/kaggle_ops.py package && python3 kaggle/kaggle_ops.py push-code`.

## Submitting

```bash
python3 kaggle/kaggle_ops.py push-models --models-dir models/<run>   # checkpoints -> dataset slotknee-models
python3 kaggle/kaggle_ops.py push-submit                             # runs the submit kernel (prints the version)
python3 - <<'PY'
import sys; sys.path.insert(0,"kaggle"); import kaggle_ops as K
K.api().competition_submit_code("submission.csv", "message", "rsna-knee-abnormality-detection",
                                kernel="felipedeleon11/slotknee-submit", kernel_version=<version>)
PY
```
The submit kernel groups checkpoints by input layout, rank-averages the groups, applies
anchor-jitter TTA on a GPU, and on a CPU fallback caps itself to 2 members without TTA so it
always finishes inside the 9 h limit.

## Local

```bash
python3 -m pytest tests/test_slots.py tests/test_slotknee.py tests/test_llm_labels.py tests/test_folds.py tests/test_slotknee_pipeline.py -q
bash scripts/smoke_slotknee.sh                                     # 48-study end-to-end, <15 min
python3 scripts/train_slotknee.py --cache cache/slots_P224_full/slots_P224 --data-dir data_subset \
  --labels data_subset/labels_external/stevenleehans/llm_labels_v4_blend.csv \
  --weights-from data_subset/labels_external/stevenleehans/llm_labels_v2.csv \
  --folds 0 1 2 3 4 --epochs 8 --bs 8 --out models/<run>
CACHE252=cache/slots_P252_full/slots_P252 CACHEG10=cache/slots_P224_g10t1_full/slots_P224_g10t1 \
  bash scripts/ablate_slotknee.sh cache/slots_P224_full/slots_P224 models/ablate_full 8 0 1
```
Long local jobs: launch under `caffeinate -ims` in their own session (`setsid caffeinate -ims
bash scripts/ablate_slotknee.sh … &` survives a terminal restart); laptop numbers are ~0.02
below T4 for the same recipe, so use them to rank arms, not as final models.

## Decision rules

- Trust an ablation delta only if it exceeds 2× the baseline seed-to-seed gap the driver prints.
- Final models: T4, `--full-fit`, the winning recipe, ≥2 seeds; main-track submission =
  multi-view rank average; efficiency-track submission = one distilled student
  (`--distill-targets data_subset/labels_external/teacher_*.csv`).
- Keep two selected submissions on Kaggle: best-AUC ensemble and the fast student.

## CURRENT STATE (2026-09-16) — read this first; the dated sections below are history

**Adopted recipe**: coverage layout `slots_P224_g10t1` (6 slots × 10 anchors, 140 mm, 224 px)
· `vit_small_patch14_reg4_dinov2.lvd142m` · self-distillation 0.5 (`teacher_g10_oof.csv`) ·
random anchor bagging 6-of-10 · **16 epochs** (adopted 2026-09-06: +0.0160 pooled vs 8, gate
0.0011). Grouped folds by scanner fingerprint. Full-anchor read-out at inference (multi-bag
closed). Noise floor: **0.0012 pooled for 8-epoch arms** (the two 8-epoch control OOFs on disk: `recipe_base_s42` 0.8650 vs `recipe_base_s1337` 0.8662, folds 0-1; gate 0.0023). The **0.0005** quoted until 2026-09-09 rested on an s42 value of 0.8657 that no OOF reproduces (see the 09-09 correction); no 16-epoch seed pair exists, so 16-epoch arms are gated with the same 0.0023 until one is measured.

**Best measured numbers**: 16-epoch 5-fold **0.8812** pooled / **0.897** macro-10 vs the 58 expert
labels (MCL 0.887, ACL 0.931); public LB **0.866** (older model). Expected for this accuracy entry:
**~0.89–0.90 LB**, higher if the forum's LB > OOF pattern holds.

| kernel | version | what | state |
|---|---|---|---|
| `slotknee-train-g10` | v18 | 16-epoch 5-fold (superseded: `train-final` v11 produced it) | not needed |
| `slotknee-train-student` | v5 | full-fit 16-epoch student; now prefers **teacher v3** (16-ep OOF) | v3-trained student pending a T4 run (the v2-trained one is in the dataset) |
| `slotknee-submit-eff` | latest | EFFICIENCY entry: `student_*_best.pt`, TTA off | **ran on a T4 09-07 09:19** — student alone, 3/3 sample studies, real submission.csv → **submit that version (efficiency track)** |
| `slotknee-submit` | latest | ACCURACY entry: **both 5-fold sets** (`*fold_*_best.pt`, student excluded), two layout groups rank-averaged per finding, TTA on — pooled OOF **0.8843** | push 09-16 → submit |
| `slotknee-ablate-g10` | **v29** | **session 4 DONE (v28, 09-08/09; gate 0.0023 from the two 8-ep controls): central +0.0038/+0.0027 ADOPT, flipswap +0.0027/+0.0015 MARGINAL, auxslot NULL, lora_r8 −0.018 REGRESS.** v29 = session 5: `central_e16 → central_flip_e16 → slotemb` (zoomm heads session 6) | **sessions 5 (v32) and 6 (v33) DONE**: central_e16 +0.0025 ADOPT; ssl_ep30/ep10 −0.078/−0.075 REGRESS; slotemb NULL; **central_flip_e16 −0.0002 vs central_e16 NULL (flip-swap closed at 16 ep); zoomm +0.0028 / +0.0016 MARGINAL** — `central_e16 → ssl_ep30 → ssl_ep10 → slotemb`; session 6: `central_flip_e16 → ssl_ep20 → zoomm` | **launch from the UI on T4 x2 now** |
| `slotknee-ssl` | **v6** | **RAN on a T4 09-08 16:27 → COMPLETE: 30 MAE epochs in 190 min, loss 0.817 → 0.436, snapshots strict-loadable** | **CLOSED 09-15**: as the encoder init of the recipe the snapshots REGRESS by 0.075–0.078 on every label (session 5) |
| `slotknee-cache-zoomm` | v1 | CPU: `COR_FS_Z80M` medial-column zoom slot (80 mm, +20 distal / 32 medial of the joint) → `slots_P224_g10t1_zm`, 14.43 GB | **v1 COMPLETE 09-08** (37 min build; 4,407 studies; joint fallback 15.5 %, side fallback 1.6 % — both pre-gates pass) → attachable to the ablate kernel |
| `slotknee-train-final` | **v13** | **DONE 09-16: the central-cache 16-epoch 5-fold, pooled OOF 0.8836** (+0.0023 vs the g10t1 5-fold 0.8812; ties it on the 58 expert labels); v11 = the g10t1 5-fold | done; both sets are in `slotknee-models` |
| `slotknee-models` | latest | **09-16**: `fold_0..4_best.pt` = the central-cache 16-ep folds (trim 0.35), `g10t1_fold_0..4_best.pt` = the coverage 16-ep folds, `student_fold_0_best.pt` | current |
| `slotknee-code` | latest | tree-identical (`src/`, `scripts/`, teachers v2+v3, `rescore_oof.py`, `aux_as_run.py`, the medial zoom centre, OOF `aux` logits, SSL defaults, `is_medial_slot` mirror rule) | current (re-pushed twice 09-08; API lists v34 ready) |

**Closed (measured, do not reopen)**: resolution · attention/MIL/pooling variants · anchor
count · bag 8 vs 6 · refined labels · per-label distillation · B/C jitter · layout ensemble ·
multi-bag inference · seed ensembling on the bagged recipe · g10t3 (fold-0 −0.004).
**Small, repeatable, adopt-for-accuracy-only**: zoomc (+0.003…+0.0045; +33 % inference).
**Alive**: the per-label slot prior (coronal-only read MCL +0.0215) — `sag_spec` in session 3
completes the picture. **Awaiting a decision**: KMAR-50K external SSL corpus.

**Architecture research (2026-09-07 literature review)**: `docs/research_architecture_20260907.md`. Read its executive summary first. Order of business: (0) submit the 16-epoch accuracy entry — the forum reports LB > OOF; (1) LoRA across all blocks; (2) hflip + medial/lateral label swap; (3) contiguous central slices; (4) aux heads; (5) slot identity inside the encoder. Decisions pending: OrthoFoundation-L eligibility, KMAR-50K licence.

**Design review (2026-09-08)** on the three levers the literature review left open (synovitis fill, SSL recipe, medial-column ROI). Adopted and applied: the synovitis fill is CLOSED (it is already in v4_blend; strengthening it hurts on gold and on pilkwang's 694 verdicts); `ssl_pretrain.py` now defaults to the S5 recipe; every OOF npz carries the per-slot aux-head logits (`aux` [N, S, 12] + `slot_names`; `scripts/aux_as_run.py` turns one slot into a pseudo-run for `compare_arms.py`); `src/slots.py` has a third zoom centre, `"medial"` (coronal bases only), and the `slotknee-cache-zoomm` CPU kernel builds it. NOT adopted (REVISE, see the 09-08 section): the `zoomm` ablate arm has no queue position yet, the SSL gate needs three sessions, Baker's silent-negative repair must transform label AND teacher, box-aux localisation is a later session. Anchor correction: on the 16-epoch model LatMen (0.758 vs 0.879) is the larger gap, not MCL (0.887).

**Launch order this quota week**: submit-eff (15 min) → train-g10 (8 h) → ablate session 3
(8 h) → ssl (2 h). GPU launches are UI-only (API lands on a banned P100); a GPU-enabled API
push doubles as a free quota check.

## THE CURRENT RECIPE (2026-08-25) — deployed and verified on Kaggle

Every element below is MEASURED, gated, and live in the deployed kernels (verified by pulling
each kernel back and inspecting its source, not by trusting the local tree):

| element | evidence | flag |
|---|---|---|
| coverage layout g10t1 (6 slots x 10 anchors, 140 mm, 224 px) | +0.0325, the campaign's biggest win | cache `slots_P224_g10t1` |
| **reg4 backbone** (DINOv2-S with registers) | +0.0220 pooled, ACL +0.0617 | `--backbone vit_small_patch14_reg4_dinov2.lvd142m` |
| **self-distillation** | +0.0185 pooled; leak caveat below | `--distill-targets teacher_g10_oof.csv --distill-weight 0.5` |
| **random slice bagging** | **+0.0090 vs the champion on THIS layout**; ACL +0.032, MCL +0.019, LatMen +0.012 | `--anchor-bag 6` |
| aux slot loss 0.2 | removing it costs **−0.064** | default |
| grouped folds (scanner fingerprint) | random folds inflate CV ~0.053 | default |
| fast decode + index reuse | pixel-identical, ~78 → ~60 min | `--decoder dicomsdl` |
| 3-shift TTA (accuracy entry only) | +0.0044; nets −0.0098 on efficiency | `SK_TTA=1` / `0` |

**Caveat that must travel with this recipe**: a distilled run's OOF is INFLATED (the teacher
carries cross-fold information), so it is NOT a gate metric — compare recipes on non-distilled
runs or on the public LB. The hidden test set is in no fold, so distillation is sound for a
SUBMISSION.

### To run it

| kernel | what it produces | how |
|---|---|---|
| `slotknee-train-g10` v13 | the model — 5 folds of the full recipe (~8 h) | UI → Settings → **GPU T4 x2** → Save & Run All |
| `slotknee-ssl` v3 | SSL-pretrained encoder (~2 h), then re-train | same |
| `slotknee-ablate-g10` v8 | gates zoomc → zoomj → tau3 → cutrot → **g20_bag6** | same |
| `slotknee-submit` v11 | submission.csv from pushed checkpoints | same, then **Submit to Competition** |

After a training run: `push-models --models-dir <run>` then run the submit kernel.
Efficiency entry: same submit kernel with `SK_TTA=0` (worth ~+0.010 on that track).

**Expected**: ~0.875 on the public LB, against 0.866 today. The field reports 0.929-0.947.

### 2026-08-29 — measured, not recalled: the API cannot launch these runs

`slotknee-train-g10` v13 was pushed with `enable_gpu: true` to test whether the
accelerator choice can be made from the API.  It cannot.  The run was assigned a
**Tesla P100 (sm_60)**, which Kaggle's own PyTorch build no longer compiles for
(it supports sm_70+), and `check_gpu()` aborted it in about a minute:

    GPU: Tesla P100-PCIE-16GB sm_60
    The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120

The SDK's `ApiSaveKernelRequest` exposes only `enable_gpu` / `enable_tpu` — there
is no accelerator-type field at all (kaggle 1.7.4.5).  The comment in
`check_gpu()` claiming the UI choice "persists across API pushes" is **not
supported by this test**: the push landed on a P100 regardless.  Treat every GPU
run as a UI action.  The fast guard means a mistaken API push costs ~1 minute of
quota, not a wasted session.

### bag8 — NULL, keep bag6

`--anchor-bag 8` vs `--anchor-bag 6`, same cache, same folds, same seed, 8 epochs:

| arm | pooled AUC | delta | focus | verdict |
|---|---|---|---|---|
| bag6_s42 (base) | 0.8626 | - | 0.7938 | - |
| bag8_s42 | 0.8631 | **+0.0005** | **-0.0022** | NULL (noise 0.0033) |

Sampling 8 of 10 anchors is indistinguishable from 6 of 10, and the focus metric
(ACL/MCL/LatMen) is marginally *worse*.  **Keep `--anchor-bag 6`** — it is also
25% cheaper per epoch.  This is evidence that the variety bagging supplies is
already saturated at 6-of-10, which weakens (does not close) the case for the
revived `g20_bag6` arm: drawing 6 from a pool of 20 is still a different
mechanism from drawing 8 from 10, but the expected payoff just got smaller.
Do not spend a session merging the G20 cache halves ahead of the queued
augmentation arms.

**Caveat on an earlier number**: comparing bag6/bag8 against `r4distill_s42`
yields a headline +0.0245, but that baseline ran at ~6 min/epoch versus ~36 for
bag6 — a 6x gap that matches the old 18-image layout, not `slots_P224_g10t1`.
That delta conflates the coverage layout with bagging and must not be quoted as
the bagging effect.  The layout-matched figure remains **+0.0090**.

### 2026-08-29 — Kaggle-only. The ablate kernel was measuring a model nobody trains

`slotknee-ablate-g10` (v9) had a defect that invalidated every arm it would have
produced: it passed **no** `--backbone`, **no** `--distill-targets` and **no**
`--anchor-bag`, and loaded `vit_small_patch14_dinov2.lvd142m.safetensors`.  With
argparse defaulting the backbone to plain DINOv2, each arm was measured on the
pre-campaign baseline — not on the adopted recipe.  Any delta it returned would
have hit the same transfer problem already documented twice above.

Fixed: a `base` list (reg4 + `--distill-weight 0.5` + `--anchor-bag 6`) is now
threaded into every arm's command, with the reg4 safetensors as
`--pretrained-path`.  Arm `extra` flags are appended **after** `base`, so any arm
can still override a base setting (argparse takes the last occurrence).
`--epochs`/`--bs` now honour `SK_EPOCHS`/`SK_BS` instead of being hardcoded to 8.

**`recipe_base` runs first and is not optional.** Changing the recipe orphaned the
old cross-session reference (`slotknee-train-g10` v5 folds 0-1), so an in-session
control is the only thing that makes the other arms interpretable.

New arm order (~2.5 h each, ~3 fit a 9 h session):

    recipe_base -> nobc -> zoomc -> nogdrop -> noshift -> zoomj -> gold32
                -> g10_cutrot -> g20_bag6

`nobc` is promoted to the first real arm: brightness/contrast jitter has been on
since the campaign began, was never chosen or measured, and is the strongest
theoretical suspect on MRI.  `g20_bag6` is demoted to last on the bag8 evidence.

Local training is stopped and the laptop is out of the loop: `gold32` was killed
after 14 h wall clock for 11 epochs (~377 min/epoch on fold 1, thrashing against a
full disk) and now runs as a Kaggle arm instead.

### 2026-08-29 — audit: two real bugs, and what was NOT wrong

**Bug 1 — the silent round-robin fold fallback (`scripts/train_slotknee.py`).**
Grouped fold assignment was wrapped in a bare `except:` that quietly substituted
`np.arange(n) % n_folds` on ANY failure.  Ungrouped folds let scanner identity
straddle the split and inflate CV by ~0.053, so a broken run scored **better**
than a correct one with nothing in the log to say so — and the bare form also
swallowed `KeyboardInterrupt`/`SystemExit`.

The fix took three attempts and the first two were wrong, which is worth recording:

1. *Raise on any failure* — broke the pipeline test.  A 6-study `--max-studies`
   smoke run genuinely cannot fill 5 stratified folds; that is a capacity limit,
   not corruption.
2. *Decide by class counts* — wrong in BOTH directions.  On a healthy 200-study
   frame a rare stratum made `too_small` true, so a corrupt kept-fold value would
   have been excused: precisely the failure the guard exists to catch.
3. *Dispatch on exception type* — correct.  `src/folds.FoldIntegrityError`
   (subclass of `ValueError`, so no caller changes) is raised for the two real
   integrity checks — missing `group` column, kept folds out of range — and is
   re-raised by the trainer.  sklearn's capacity errors and the "only N groups"
   check stay plain `ValueError` and are absorbed.  **Every** fallback now prints
   a banner: the original bug was the silence, not the fallback.
   `--allow-ungrouped-folds` overrides.  Four regression tests in
   `tests/test_folds.py` pin the distinction.

**Bug 2 — non-reproducible submissions (`scripts/infer_slotknee.py`).**
Multi-bag inference drew anchors with `torch.randperm` and the script seeded
nothing at all, so the same checkpoint scored differently on every run and an LB
number could not be reproduced.  Now uses a seeded generator (`--seed`, default
42).  `--bag` without `--bags > 1` also silently no-opped; it now says so.

**What the audit did NOT find.** Zero dead top-level symbols across `src/` and
`scripts/`, zero never-read CLI flags out of 91, zero syntax errors, duplicate
defs, mutable default args or `is`-with-literal comparisons.  The waste was in
artifacts, not code: 40 GB of local caches (all COMPLETE on Kaggle) were removed,
and `kaggle/.logs/` was added to `.gitignore` — it was untracked AND unignored
(line 38's `logs/` does not match `.logs`), leaving 15 GB of cache and the
trained fold checkpoints one `git add -A` from being committed.

Suite: **478 passed** (474 + 4 new).  Code dataset re-pushed; `slotknee-ablate-g10`
is **v11**, verified by pull-back.

**Verification gap, stated plainly**: the code *dataset* could not be pulled back
to confirm byte-for-byte.  In kaggle 1.7.4.5 `dataset_download_file(s)` returns
404 and `dataset_list_files` caps at 20 entries.  `push-code` reported success and
the packaged payload byte-matched the working tree before upload, but that is one
step weaker than the kernel verification, where `kernels_pull` works.

### 2026-08-29 — THE FULL RECIPE TRAINED, 5 folds, on a T4

`slotknee-train-g10` completed: **Tesla T4 sm_75, 244 min, folds 0-4, 8 epochs**,
running the adopted recipe end to end —

    --backbone vit_small_patch14_reg4_dinov2.lvd142m
    --distill-targets teacher_g10_oof.csv --distill-weight 0.5
    --anchor-bag 6

Pooled OOF over all 4407 studies, against the previously pulled checkpoints on
identical folds (uid-overlap 1.0000):

| run | pooled OOF | per-fold | delta | focus |
|---|---|---|---|---|
| previous checkpoints | 0.8531 | .845/.862/.854/.846/.858 | - | 0.7774 |
| **this run** | **0.8643** | .861/.869/.869/.855/.866 | **+0.0112** | **+0.0196** |

All twelve findings improved, none regressed, and the three weakest carry the
gain: ACL +0.0208, MCL +0.0193, LatMen +0.0187.

**Do NOT read +0.0112 as a clean gate.** This run is distilled, and a distilled
run's OOF is inflated by cross-fold teacher information (see the caveat above).
The baseline's own recipe could not be confirmed — the local log under
`kaggle/.logs/slotknee-train-g10/` belongs to a *different*, failed P100 run than
the checkpoints beside it. So the honest reading is: the recipe trained
end-to-end and scores 0.8643 OOF; how much of the +0.0112 is real improvement
versus teacher leakage is **not** established by this comparison. The submission
itself is unaffected — the hidden test set is in no fold.

Attention entropy sat at 0.999-1.000 for every epoch of every fold
(`[ATTENTION COLLAPSED -> mean pooling]`). That is the **expected, documented**
behaviour — diagnosed 2026-08-24 and vindicated by the literature review; it is a
closed family, not a new defect.

Checkpoints pushed: `felipedeleon11/slotknee-models` now holds
`fold_0..4_best.pt` from this run. Next step is `slotknee-submit` (its stored
metadata has `enable_gpu: false`, so set the accelerator in the UI), then
**Submit to Competition**.

### 2026-08-29 — CPU pre-flight: verified cache names (no more guessing)

`check_gpu()` now runs AFTER input discovery in all four GPU kernels
(`train-g10`, `train-final`, `ablate-g10`, `ssl`). This matters because a CPU push
is the *only* run the API can produce, and it used to print nothing but
"No GPU". It now validates every mount before aborting, which immediately paid off:

| cache dir (verified on Kaggle) | source kernel |
|---|---|
| `slots_P224_g10t1` | `slotknee-cache-g10t1` |
| `slots_P224_g10t1_z100j` | `slotknee-cache-zoomj` |
| `slots_P336_g4t1` | `slotknee-cache-p336g4` |
| `slots_P224` (old v2 layout) | `slotknee-cache-builder` |

So the ablate arms resolve: `zoomj` asks for `slots_P224_g10t1_z100j` — **correct**.
`zoomc` asks for `slots_P224_g10t1_zc`, which matches the verified local cache and a
COMPLETE Kaggle kernel. Only **`g20_bag6` will skip**: the halves are published as
`slots_P224_g20t1_partA` / `partB`, not the `slots_P224_g20` the arm requests, so
`scripts/merge_slot_cache.py` must run first. It is last in the order, so a 9 h
session (≈3 arms) never reaches it.

Two kernels were CANCEL_ACKNOWLEDGED (`submit` v12, `train-final` v7), both when
several kernels were pushed at once — push one at a time.

### 2026-09-02 — AUC audit toward 0.90: see docs/auc_audit_20260902.md

New 5-fold model re-scored against the 58 expert labels: **macro-10 0.871** (champion 0.851).
Model-limited findings unchanged in shape — MCL 0.739, LatMen 0.724, MedMen 0.833, PF OA 0.801,
ACL 0.890 — four of five coronal joint-line. Ranked plan: zoomc → per-label distillation
(new arm, weights `0.1,0.1,0.1,0.1,0.7,0.5,0.2,0.7,0.5,0.5,0.7,0.7`) → layout ensemble → SSL
(fix its cache to g10t1 first) → g10t3.

**Closed: refined labels.** `llm_labels_v6_refined` scores identically to `v4_blend` against
expert gold (0.893/0.909); the `r4refined` arm's +0.011 was measured against its own altered
targets. `compare_arms.py` scores each run against the `y` stored in its own OOF, so two runs
trained on DIFFERENT label files are never comparable through it — score both against gold.

Ablate kernel arm order is now: recipe_base → zoomc → distil_perlabel → nobc → nogdrop →
noshift → zoomj → gold32 → g10_cutrot → g20_bag6.

### 2026-09-02 — implemented and launched from the audit

| what | where | state |
|---|---|---|
| per-label distillation arm (`distil_perlabel`) | `slotknee-ablate-g10` **v16**, arm 3 | deployed, verified by pull-back |
| sequence-specialist arms (`cor_spec`, `sag_spec`) | same kernel, arms 7-8 | gate the per-label slot-prior idea |
| run-time cache **part merging** in the arm loop | same kernel | unblocks `g20_bag6` (now asks for `slots_P224_g20t1`) and `g10t3` |
| `g10t3` arm (T=3 triplets on the coverage layout) | same kernel, last | skips cleanly until its cache is attached |
| SSL kernel switched to the coverage cache | `slotknee-ssl` **v5**, `kernel_sources = slotknee-cache-g10t1` | pre-flight confirmed: resolves `slots_P224_g10t1` |
| **g10t3 cache build**: 39.8 GB in THREE parts | `slotknee-cache-g10t3a/b/c` v1 (CPU) | **RUNNING** — 1469 studies each, ~13.3 GB each, ~8 h |

Why three parts: 4407 × 6 × 10 × 3 × 224² B = 39.8 GB; two halves would be 19.9 GB each,
over Kaggle's 20 GB output cap. `kaggle_ops.py` knows them as `push-cacheg10t3a/b/c`.

**When the builders finish** (next session): add `felipedeleon11/slotknee-cache-g10t3a`, `-b`,
`-c` to `kaggle/ablate_kernel_g10/kernel-metadata.json` `kernel_sources`, push, and the `g10t3`
arm merges the parts onto /kaggle/temp itself. Nothing else to do.

Ablate arm order: recipe_base → zoomc → distil_perlabel → nobc → nogdrop → noshift → cor_spec →
sag_spec → zoomj → gold32 → g10_cutrot → g20_bag6 → g10t3. ~3 arms fit a 9 h T4 session; the
first three are the ones the audit ranked highest. **GPU launch is still a UI action.**

### 2026-09-02 — g10t3 cache BUILT (three parts, 16-20 min each) and attached

| part | studies | failures | size | wall |
|---|---|---|---|---|
| `slotknee-cache-g10t3a` | 1469 (+ 3-study test split) | 0 | 12.38 GB | 20.0 min |
| `slotknee-cache-g10t3b` | 1469 | 0 | 12.36 GB | 16.0 min |
| `slotknee-cache-g10t3c` | 1469 | 0 | 12.36 GB | 16.4 min |

37.1 GB total, exactly 4407 × 6 × 10 × 3 × 224² bytes. Attached to `slotknee-ablate-g10`
(**v17**, verified by pull-back); the `g10t3` arm is now **arm 5**, right after `nobc`, so it
opens session 2. The kernel merges the parts onto /kaggle/temp itself.

**Correct the campaign's cache-build estimate.** With dicomsdl and 4 workers a CPU kernel
builds at **1.4-1.6 studies/s** — a full 4407-study cache in ~45-50 min, not the ~500 min that
older notes assume. Cache experiments are cheap now; the 20 GB *output* cap, not time, is the
constraint, and the run-time merge removes it.

Reading a completed builder without downloading its payload: `kernels_output` pulls every
shard (13 GB) and times out; use `list_kernel_session_output` for the file URLs and fetch only
`cache_builder_report.json` / `train_index.json` plus `response.log`.

### 2026-09-03 — ablate session 1 on a T4: 5 arms, 495 min, NONE adopts

In-session control `recipe_base` = **0.8657** pooled (folds 0-1, 1760 studies) *[corrected 2026-09-09: the OOF of this run, re-fetched from kernel version 13, scores **0.8650** with `compare_arms.py`; 0.8657 does not reproduce under any scoring rule]*, reproducing the
5-fold run's folds 0-1 (0.8607/0.8693). Arms take **~95 min** on a T4 (5.8 min/epoch × 8 × 2
folds + cache copy), not the 2.5 h assumed — **five arms fit a session**. The run-time part
merge worked first time: 3 parts → 4407 studies in 460 s.

| arm | pooled Δ | focus Δ | verdict | per-label note |
|---|---|---|---|---|
| zoomc | **+0.0045** | +0.0016 | NULL | MedMen +0.0137, ACL +0.0083, MedOA +0.0080, PF OA +0.0076, **MCL −0.0067** |
| distil_perlabel | +0.0008 | +0.0014 | NULL | MedMen +0.0084, ACL +0.0049, Fracture −0.0040 (weight 0.7 hurt it) |
| nobc | −0.0006 | −0.0015 | NULL | brightness/contrast jitter is harmless — closed |
| g10t3 | −0.0152 | −0.0229 | REGRESS* | *fold 1 truncated at 3 epochs by the cap; **fold 0 alone −0.0041** |

**Readings.**
* zoomc sits between 1× and 2× noise and moves the *wrong* way on MCL — the coronal zoom
  helps the medial compartment and ACL, not the finding with the largest gap. At +33 %
  inference cost it is efficiency-negative and, on this evidence, not an accuracy adopt
  either. A second seed would say whether +0.0045 is real; it is not worth a 5-fold on its own.
* Per-label distillation and the B/C jitter are **closed**: the hypotheses were reasonable
  and the measurements are clean nulls.
* g10t3 is **not cleanly measured** (truncated fold), but fold 0 leans negative; through-plane
  context did not help on this evidence. Re-run only if a session has spare room.
* Three results landed in the ambiguous band, and the 0.0033 noise floor was measured on the
  OLD layout. **The most valuable next arm is a second seed of `recipe_base` on this layout**:
  it pins the floor for every future gate and, via `select_ensemble.py`, tests whether seed
  ensembling adds anything now that bagging decorrelates the inputs (the 0.953 seed
  correlation was measured on deterministic inputs).

The vs-expert-gold column has n=22 in these folds (2-4 positives for MCL) and is not
interpretable arm-to-arm at that size; it is recorded in the session output only.

**Consequence for the 0.90 plan**: the top three ranked levers are spent. What remains
untested is structural — SSL (`slotknee-ssl` v5, ready), the missing-modality handling, seed
ensembling on the bagged recipe, and the flat-bag redesign — plus the label-handling axis the
4th-place competitor names as decisive.

**Layout ensemble measured (same session): NULL.** `select_ensemble.py` over recipe_base /
zoomc / distil_perlabel / nobc (1760 common studies), honest held-out scoring: best single
0.8697 (zoomc) → greedy ensemble 0.8697, **−0.0000**. Two genuinely different layouts and
rank-averaging them adds nothing out of sample. The "+0.005-0.010 from a layout ensemble" in
the audit is withdrawn; the submit kernel's multi-group averaging stays as a mechanism, but
there is no measured pair worth feeding it.

**Session 2 queue deployed** (`slotknee-ablate-g10` **v19**, verified by pull-back; the kernel
now honours a per-arm `--seed` and names the output dir by it):

    recipe_base(s1337) → zoomc(s1337) → cor_spec → sag_spec → nogdrop → noshift → zoomj
    → g20_bag6 → gold32 → g10_tau3 → g10_cutrot → g10t3

Five fit a session. The first two pin the noise floor on this layout and settle zoomc; the
next two decide the slot-prior idea. Closed arms (nobc, distil_perlabel) are removed so a
re-launch does not re-measure them.

### 2026-09-03 — EFFICIENCY ENTRY prepared: one full-fit distilled student

The exchange rate makes this the efficiency-track play (0.01 AUC ≈ 12 min over 1,300
studies; ensemble 1.18 s/study vs one student 0.08 s/study = +0.0199 AUC-equivalents of
headroom). Everything is wired; it needs one T4 session.

| piece | where | state |
|---|---|---|
| **teacher v2** = the 5-fold adopted-recipe run's OOF, probability-averaged | `labels_external/teacher_g10_v2.csv` (4407 studies; rank-corr 0.937 with the old teacher) | in the code dataset (pushed) |
| **student kernel**: `--full-fit`, reg4 + bag6 + distil 0.5 from teacher v2, 10 epochs, one checkpoint | `slotknee-train-student` **v1** (kaggle/train_kernel_student) | pushed, verified by pull-back; CPU pre-flight in progress |
| **checkpoint selection** in the submit kernel: `SK_CKPT_GLOB` | `slotknee-submit` **v16** | pushed, verified by pull-back |

**Why `SK_CKPT_GLOB` had to exist first**: `push-models` rebuilds the models dataset from
scratch, so pushing the student alone would have *deleted* the five fold checkpoints from
`slotknee-models`. Push BOTH sets into one version, with a prefix on the student:

    python3 kaggle/kaggle_ops.py push-models --models-dir <5-fold dir> --models-dir student=<student dir>

then each entry selects its own set:

| entry | kernel env | members | TTA |
|---|---|---|---|
| accuracy | `SK_CKPT_GLOB=fold_*_best.pt` | 5 folds | on (3 shifts, +0.0044) |
| efficiency | `SK_CKPT_GLOB=student_*_best.pt` `SK_TTA=0` | 1 student | off |

There is no OOF from a full fit; the student is judged only on the public LB, efficiency
track. It "wins" if it scores within 0.0199 of the accuracy entry's AUC.

**To run**: `slotknee-train-student` → GPU T4 x2 → Save & Run All (~75-90 min). Pull the
checkpoint, `push-models` as above, run `slotknee-submit` with the efficiency env, submit.

### 2026-09-03 — GPU quota EXHAUSTED: "Maximum weekly GPU quota of 30.00 hours reached"

An API push of `slotknee-ablate-g10` with `enable_gpu: true` was refused before creating a
version. This week's 30 h went to the 5-fold training run (244 min, T4), ablate session 1
(495 min, T4), and earlier sessions. **No GPU kernel can run — from the API or the UI — until
the weekly reset** (Kaggle shows the countdown on the kernel's Settings → Accelerator panel;
the API does not expose it). CPU work (cache builds, pre-flights, CPU submits) is unaffected.

Two useful side facts. A GPU-enabled API push is a **free quota check**: when quota is
exhausted it fails fast with this exact message and burns nothing; when quota exists it
starts a P100 run that the guard aborts in ~1 min. And every GPU hour now has a price: at
~95 min per ablate arm, a session is 5 arms ≈ 8 h ≈ **a quarter of the week**. Order the
queue accordingly — it already is.

**When the quota resets, in this order:** (1) `slotknee-ablate-g10` session 2 (noise floor +
zoomc seed 2 + the slot-prior gate, ~8 h); (2) `slotknee-train-student` (~1.5 h) → efficiency
entry; (3) `slotknee-ssl` (~2 h). That is ~11.5 h of the next 30, leaving room for the
5-fold re-train any adopted arm will need.

### 2026-09-03 — two more levers wired for session 2 (no GPU needed to prepare)

**1. Longer training WITH bagging — `recipe_e16`, arm 3.** `ep14` was NULL (+0.006), but it
was measured with the model re-seeing the identical 60 images every epoch. Bagging draws a fresh
6-of-10 per epoch, so the argument that closed longer schedules no longer applies. 16 epochs
costs two arm slots (~190 min), zero at inference.

**2. Multi-bag inference — `scripts/rescore_oof.py` + a post-session step.** The recipe trains
on random 6-of-10 anchor bags but validates and submits on all 10. Two reasons the read-out
matters: it is a train/inference distribution mismatch, and `group_embed` is indexed by
*position* (`group_embed[:G]`), so with bag 6 positions 6-9 are **never trained** (they stay
at their 0.02-std init, decayed toward 0 — small, but every full-anchor read-out adds four
untrained offsets per slot). Averaging N independent 6-bags at inference avoids both and is
free ensembling from one set of weights, at N×0.6 forward-equivalents.

The tool re-scores a fold checkpoint on exactly the studies the fold held out (uids and labels
from the OOF npz — no fold logic re-derived), reports macro AUC for both read-outs and the
per-label delta, and writes an OOF-style npz. Tested end to end on the micro fixture
(`tests/test_rescore_oof.py`). The ablate kernel now runs it on the session's `recipe_base`
control at the tail of the session (6-of-10 × 4 bags, ~10 min on a T4; skipped under 25 min
left) and writes `rescore_report.json`. **If it adopts, the same read-out goes into the submit
kernel as `--bag 6 --bags N` — accuracy entry only** (N×0.6 forwards is efficiency-negative
unless the gain beats the runtime exchange rate).

Deployed: `slotknee-ablate-g10` **v20**, verified by pull-back; `scripts/rescore_oof.py` is in
the code dataset. Session-2 queue: recipe_base(s1337) → zoomc(s1337) → **recipe_e16** →
cor_spec → sag_spec → … → [rescore recipe_base].

**Awaiting a decision — KMAR-50K.** 1,444 multi-parameter knee MRI scans, CC BY 4.0, ungated,
~10 GB (Mendeley 10.17632/xw7mrg7ntg.6). The only clean external corpus; it would double the
SSL pretraining set with different scanners. It is external data and a download, so it is NOT
fetched. If authorised: fetch → convert to a slot cache with the existing builder → attach to
`slotknee-ssl` alongside g10t1.

### 2026-09-06 — ablate session 2 (T4, 4 arms): 16 EPOCHS ADOPTED, +0.0160

Control `recipe_base_s1337` = 0.8662 pooled (folds 0-1). The session's first job was the noise
floor on THIS layout, and it changed the gate for everything after it:

**Noise floor = 0.0005** (seed 42 vs seed 1337 of the recipe: 0.8657 vs 0.8662, same folds). *[corrected 2026-09-09: the s42 OOF scores 0.8650, so the seed-to-seed spread is 0.0012 and the 8-epoch gate is 0.0023; session-2 verdicts near the old 0.0011 gate — zoomc +0.0028 / +0.0039 on the two controls — remain adopt-for-accuracy-only]*
Bagging makes seeds almost interchangeable in pooled AUC. The gate is now **±0.0011**, six
times tighter than the old-layout 0.0033 — which retroactively makes session 1's zoomc
(+0.0045) a real, small effect rather than noise.

| arm | pooled Δ | focus Δ | verdict | reading |
|---|---|---|---|---|
| **recipe_e16** (16 epochs) | **+0.0160** | **+0.0330** | **ADOPT** | MCL +0.048, LatMen +0.032, ACL +0.019; val AUC **still rising at 16** on both folds |
| zoomc (seed 1337) | +0.0028 | +0.0027 | ADOPT (small) | +0.0033 / +0.0045 on the other seed/control: repeatable, but MCL inconsistent and +33 % inference |
| cor_spec (coronal only) | −0.0096 | +0.0068 | REGRESS | yet **MCL +0.0215**, ACL +0.009, MedMen +0.009; Baker's −0.093 (invisible on coronal). The slot-prior idea is alive for MCL/menisci |
| multi-bag inference (rescore of control) | −0.0015 / −0.0020 | | **closed** | full-anchor read-out stays |
| seed ensemble (s42 + s1337) | +0.0007 held-out | | NULL | 4-member pool with both zoomc seeds +0.0006; with e16 in the pool the selector ships **e16 alone** |

**Why e16 is the biggest win since coverage.** `ep14` was NULL on fixed inputs because the
model re-saw the same 60 images; bagging turns every epoch into new views, so the schedule that
looked saturated at 8 is now clearly under-trained — the curves have not flattened at 16.
Zero inference cost. The three weakest findings take the three biggest gains, exactly the
pattern of bagging itself.

**Adopted everywhere**: `slotknee-train-g10` (5 folds × 16 ep, `LIMIT_MIN` 500 — ~465 min, tight
under the 540-min cap; if fold 4 truncates, re-run it alone with `SK_FOLDS=4`),
`slotknee-train-final`, `slotknee-train-student` (full fit, 16 ep, ~2 h), and the ablate
kernel's base (every future arm measures on the 16-epoch recipe; control = `recipe_e16_s42`).

**Session 3 queue**: `recipe_e24` (where is the plateau? ~285 min, the most valuable single
measurement now) → `sag_spec` (~70 min; completes the plane picture for a per-label slot prior)
→ `zoomc` at 16 ep (do the two adopts stack?) → hygiene arms. Multi-bag rescoring is opt-in
(`SK_RESCORE=1`).

**Next launches, in order**: (1) `slotknee-train-g10` — the 16-epoch 5-fold for the accuracy
entry (~8 h); (2) `slotknee-ablate-g10` session 3 (~8 h); (3) `slotknee-train-student` (~2 h);
(4) `slotknee-ssl` (~2 h). ~20 h of the 30.

**Revised expectation**: 0.8822 pooled on 2 folds → ~0.88 five-fold OOF → **~0.89-0.90 public
LB** with the accuracy ensemble, before e24 / zoomc stacking. First time 0.90 is in reach on
measured evidence rather than priors.

Deployed versions carrying the 16-epoch recipe (all verified by pull-back): `slotknee-train-g10`
**v18**, `slotknee-train-student` **v3**, `slotknee-ablate-g10` **v23** (session-3 queue),
`slotknee-train-final` **v9**. Label-independent check on the 22 gold studies in folds 0-1:
recipe_e16 macro-10 **0.942** vs 0.912/0.913 for the two 8-epoch seeds (MCL 1.000, LatMen 0.812,
MedMen 0.909) — same direction as the pooled result, wide CIs at n=22.

### 2026-09-07 — first T4 launches after the reset: student trained, train-final exposed a bug

**`slotknee-train-student` — SUCCESS.** T4, 98 min, full fit on all 4,407 studies, 16 epochs,
reg4 + bag6 + distil 0.5 from `teacher_g10_v2.csv` (confirmed from the command line; no
fallback WARN). Output `models/slotknee_student/fold_0_best.pt`. Its logged "val" is in-train and
means nothing; the student is judged on the LB, efficiency track.

**`slotknee-train-final` — FAILED at fold 0, and it was a real bug**, latent since the kernel
was written and hidden because it had never run on a GPU: it hardcoded
`weights/vit_small_patch14_dinov2.lvd142m.safetensors` (plain DINOv2) while its RECIPE passes
`--backbone vit_small_patch14_reg4_dinov2.lvd142m`. The register variant's pos_embed has a
different token count, so timm's `resample_abs_pos_embed` died with
`shape '[1, 37, 37, -1]' is invalid`. Fixed in **v11** (weights derived from the recipe's
`--backbone`, as `train-g10` already does), verified by pull-back. `train-g10`, the student and
the ablate kernel were never affected (each derives or names the reg4 file explicitly).

**Models dataset now carries both entries** (`push-models` with two dirs):

| files | set | entry | submit env |
|---|---|---|---|
| `fold_0..4_best.pt` | 8-epoch 5-fold (the run of 08-29) | accuracy (interim) | `SK_CKPT_GLOB=fold_*_best.pt` |
| `student_fold_0_best.pt` | 16-epoch full-fit student | **efficiency** | `SK_CKPT_GLOB=student_*_best.pt SK_TTA=0` |

The 8-epoch set is a placeholder: the 16-epoch 5-fold (`slotknee-train-g10` v18, ~8 h)
supersedes it and is the accuracy entry worth submitting. It has not been launched yet.

**Retrieving older session outputs.** `list_kernel_session_output` only shows the LATEST
session, so a CPU pre-flight hides a completed run's files. `ApiDownloadKernelOutputRequest`
takes `version_number` + `file_path`: the 08-29 checkpoints were recovered from version 15 this
way. Recorded in memory too.

**Correction to the table above — the UI cannot pass environment variables.** Save & Run All
has no env panel, so `SK_CKPT_GLOB` / `SK_TTA` could only ever be set by API pushes, which land
on a P100. Each entry is therefore its own kernel with the choice baked into the defaults:

| kernel | default checkpoint glob | TTA | entry |
|---|---|---|---|
| `slotknee-submit` **v17** | `fold_*_best.pt` (the 5 folds; the student is never mixed in) | on | accuracy |
| `slotknee-submit-eff` **v1** | `student_*_best.pt` (the one student) | off | **efficiency** |

Same code path; `SK_*` still override for API runs. `kaggle_ops.py push-submit-eff` pushes the
efficiency kernel. **To enter the efficiency track now**: `slotknee-submit-eff` → GPU T4 x2 →
Save & Run All → Submit to Competition (efficiency). The accuracy kernel should wait for the
16-epoch 5-fold (`slotknee-train-g10` v18) and a `push-models` of its checkpoints.

### 2026-09-07 — SESSION 4 deployed: three research levers implemented and queued

From `docs/research_architecture_20260907.md` (literature review). All tests green (486);
code dataset re-pushed; `slotknee-ablate-g10` **v24** verified by pull-back.

| lever | implementation | arm |
|---|---|---|
| **exact laterality mirror + medial/lateral label swap** (aug; the winners' "hflip + L/R swap") | `--flip-swap p` in `train_slotknee.py`: COR/AX columns flipped, SAG anchor order reversed — the same transform the cache applies to right knees — and the four medial/lateral labels (+ weights, + distillation targets) permuted. A plain hflip would be label-corrupting; this one is exact and an involution (tests pin it against `src.slots.apply_laterality`). Off by default. | `flipswap` (8 ep, `--flip-swap 0.5`) |
| **mirror TTA** | `rescore_oof.py --flip-tta`: plain and mirror-averaged read-outs reported separately (`macro_full`, `macro_flip_tta`, `delta_flip`) | post-session step runs it on the `flipswap` model (`SK_RESCORE_ARM`) |
| **slot-identity auxiliary head** (aux losses on free targets: the largest measured lever in three RSNA competitions) | `SlotKneeS(aux_slotid=True)`: linear head on each per-image feature → its slot (plane × sequence), CE over present slots, `--aux-slotid W`; zero inference cost; recorded in `hparams` so checkpoints round-trip | `auxslot` (8 ep, `--aux-slotid 0.1`) |
| **central-block anchors** (this forum: 9 adjacent central slices beat 9 spread by +0.018) | new cache `slots_P224_t35_g10t1` = g10t1 with `SLOT_TRIM 0.35` (10 anchors over the middle 30 % of each stack); `slotknee-cache-g10t1-t35` building on Kaggle CPU (~45 min); `push-cacheg10t35` registered | `central` (8 ep; skips until the cache is attached) |

**Why the cheap arms run at 8 epochs.** The adopted base is 16 epochs (~190 min/arm), which
fits ~2.5 arms per session. The two 8-epoch controls on these folds (`recipe_base_s42`,
`recipe_base_s1337`; noise floor 0.0005) already exist, so gating cheap levers at 8 epochs is
valid and 2.4× cheaper: **five arms fit**. `--epochs 8` in an arm's extras overrides the base.

Queue: `flipswap → auxslot → central → recipe_e24 → sag_spec → zoomc → …`. Analysis after the
session: `compare_arms.py` of the 8-ep arms against `recipe_base_s42` + `s1337`; `recipe_e24`
against `recipe_e16_s42`; the rescore report for the TTA half of flipswap.

Still needing decisions: OrthoFoundation-L (eligibility), KMAR-50K (licence). Still needing
your click, in order: **submit-eff → train-g10 (16-ep accuracy 5-fold) → ablate session 4 → ssl**.

### 2026-09-07 — levers 1 and 4 implemented: LoRA on all blocks, slot identity inside the encoder

| lever | implementation | arm |
|---|---|---|
| **LoRA across all 12 blocks + patch embedding** (research S1) | `SlotKneeS(lora_rank=r, lora_alpha, lora_dropout, lora_mlp, lora_patch)`: `LoRALinear` wraps every attention projection (fused `qkv` or split q/k/v, and `proj`), `LoRAConv2d` wraps the patch embedding; **all base weights frozen**, LoRA + LayerNorm affines train; no frozen prefix (every block runs in the checkpointed tail, the stem runs with gradients for the patch LoRA); `param_groups` uses a uniform LR under LoRA. `lora_B` is zero-initialised, so step 0 equals the base model (tested). r=8: 0.22 M params (attention) vs 7.1 M today. Trainer: `--lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 [--lora-mlp] [--no-lora-patch]`. | `lora_r8` (8 ep, `--lr-backbone 1e-4`) |
| **slot identity inside the encoder** (research S4, MM-DINOv2) | `SlotKneeS(slot_tok_embed=True)`: zero-initialised `nn.Embedding(n_slots, 384)` added to every token of an image at the input of the first *trainable* block (block 0 under LoRA; block depth−k otherwise), fed with each image's slot index from the (B,S,G) layout; works on the cached-activation path too. Trainer: `--slot-tok-embed`. | `slotemb` (8 ep) |

Tests: `tests/test_lora_slotemb.py` (wrapping, freezing, parameter count, step-0 equivalence,
strict checkpoint round-trip, uniform LR groups, zero-init + gradient flow, CPU micro run).

Note found while reading the optimiser code: **layer-wise LR decay already existed**
(`SlotKneeS.param_groups(llrd=0.8)`: deepest trainable block at `lr_backbone`, each earlier ×0.8).
The research write-up's "no LLRD" line was wrong; the `llrd_0.75` idea is therefore moot for the
partial-FT regime and disabled by design under LoRA.

Session-4 queue is now: `flipswap → auxslot → central → lora_r8 → slotemb → recipe_e24 → sag_spec → …`
(`slotknee-ablate-g10` **v25**, verified by pull-back). Five 8-epoch arms ≈ one session; `lora_r8`
runs ~2× slower (gradients through 12 blocks), so expect the session to stop before `recipe_e24`.

Shipped: suite **490 passed**; `slotknee-code` re-pushed with the LoRA / slot-embedding / aux-head /
flip-swap code (payload byte-identical to the tree for `src/slotknee.py`, `scripts/train_slotknee.py`,
`scripts/rescore_oof.py`). The only piece still building is the central-block cache
(`slotknee-cache-g10t1-t35`); it gets attached to `slotknee-ablate-g10` on completion.

Also on the submission path: `scripts/infer_slotknee.py --flip-tta` (mirror TTA with the
medial/lateral swap-back, per anchor-shift view; needs `slot_layout.slot_names` in the checkpoint,
which every checkpoint since the coverage layout carries). **Not enabled in either submit kernel** —
it goes into the accuracy kernel only if the `flipswap` arm's post-session TTA report adopts.

Final state of the day: suite **492 passed**; `slotknee-code` re-pushed (now carries every lever incl.
`infer_slotknee.py --flip-tta`). Tests added today: `test_flip_swap.py`, `test_aux_slotid.py`,
`test_lora_slotemb.py`, `test_infer_flip_tta.py`, `test_rescore_oof.py`, and 4 fold-guard tests.

**Central-block cache built and attached** (`slotknee-cache-g10t1-t35`: 4407 studies, 0 failures,
12.37 GB, 30.8 min). `slotknee-ablate-g10` **v27** carries it; the `central` arm no longer skips.
Session-4 queue is complete: every arm has its cache. Version table update: ablate **v27**
(supersedes v25).

### 2026-09-08 — THE 16-EPOCH 5-FOLD IS TRAINED (via `slotknee-train-final` v11, T4, ~7 h)

Felipe's T4 launch of the fixed `train-final` (09-07 09:10) ran the adopted recipe over all five
folds at 16 epochs (5.4 min/epoch). Its status read COMPLETE and I only noticed today because a
CPU pre-flight of the same kernel cannot produce COMPLETE — a useful tell.

| | pooled OOF (4407) | vs 58 expert labels, macro-10 | MCL | LatMen | MedMen | ACL | PF OA |
|---|---|---|---|---|---|---|---|
| 8-epoch 5-fold (08-29) | 0.8643 | 0.871 | 0.739 | 0.724 | 0.833 | 0.890 | 0.801 |
| **16-epoch 5-fold (09-07)** | **0.8812** | **0.897** | **0.887** | 0.758 | 0.861 | **0.931** | 0.811 |

+0.017 pooled, +0.026 on expert labels; MCL +0.15 against experts — the finding the audit
called the largest model-limited gap. (Both runs are distilled from the same teacher, so the
pooled numbers share the same inflation; the expert-label column is the clean one.)

**Models dataset**: `fold_0..4_best.pt` are now the **16-epoch** folds (they replaced the
8-epoch set) + `student_fold_0_best.pt`. `slotknee-submit` (accuracy, `fold_*`, TTA on) needs
no change — **run it on a T4 and submit**. `slotknee-train-g10` v18 is no longer needed for this
recipe. Cosmetic: the run dir is named `g10t1_reg4_distil_bag6_e16_e16_s42` (suffix doubled by
the rename); harmless.

**Expected LB** for this entry: ~0.89–0.90 by the OOF→LB offset seen so far, higher if the
forum's "LB > OOF" pattern holds for us too.

Per-label gate, 16-epoch vs 8-epoch 5-fold on identical folds and all 4407 studies (noise 0.0005):
**+0.0169 pooled, every one of the 12 labels ADOPT** — MCL +0.045, LatMen +0.034, ACL +0.025,
MedMen +0.022, Contusion +0.012, LatOA +0.012, PF OA +0.011, Baker's +0.011, Synovitis +0.010,
Effusion +0.008, MedOA +0.007, Fracture +0.007. Focus (ACL/MCL/LatMen) +0.035.

**Teacher v3** (`labels_external/teacher_g10_v3.csv`) = the 16-epoch 5-fold OOF, probability-averaged
(rank-corr 0.969 with v2). `slotknee-train-student` **v5** prefers v3 → v2 → original and prints
which it used. The student currently in `slotknee-models` was distilled from v2 (8-epoch OOF);
re-running the student on a T4 (~2 h) with v3 is the better efficiency entry. The 5-fold recipe
itself keeps `teacher_g10_oof.csv` (the adopted recipe was measured with it; a second
self-distillation round is the research's "round 2 rarely helps" case and would need its own gate).

**Launch order now**: (1) `slotknee-submit` on T4 → Submit (accuracy, 16-epoch folds — ready);
(2) `slotknee-train-student` v5 on T4 → `push-models` with both dirs → `slotknee-submit-eff` →
Submit (efficiency); (3) `slotknee-ablate-g10` v27 session 4; (4) `slotknee-ssl`.

### 2026-09-08 — design review of the three open levers: what was adopted, applied and gated

A design review took the three levers the 09-07 literature review left open and decided per change.
Rule: ADOPT only when an independent re-check could not refute it. Everything below was applied
today; every number was computed once and reproduced independently.

**1. Synovitis-from-effusion fill — CLOSED (0/3 refuted).** The forum's 0.678 → 0.790 is exactly
reader A → v2 (`fill_silent_synovitis`, `src/llm_labels.py`) on our 58 gold rows (0.6780 → 0.7903), and
v2/v4 is the file we already train on. Strengthening it: k = 0.5 changes no ranking; k ≥ 0.75 or
syn := eff is −0.05 on gold and −0.055 on pilkwang's 694 YES/NO verdicts; the model's OOF synovitis
against such labels drops 0.873 → 0.818 from the label change alone. Docs corrected (research row 6,
S2 fact 3); no arm.

**2. SSL recipe (0/3 refuted) — baked into `scripts/ssl_pretrain.py` defaults**, so the unchanged
`slotknee-ssl` v5 inherits it through the code dataset: `--blr 1e-4` (was the from-scratch MAE
1.5e-4), `--layer-decay 0.8` (BEiT ids: stem 0, block i → i+1, decoder/norm → depth+1; 28 param
groups for ViT-S, 14 distinct LRs, sorted keys so `--resume` is deterministic), `--snapshot-epochs
10 20 30` (`encoder_epNN.safetensors`, never overwritten, listed in `log.json["snapshots"]`),
`log.json` records blr/layer_decay/mask_ratio, and `verify_export` now also proves the **T=1** load
(the adopted cache is g10**t1**: patch-embed = RGB-sum, every other tensor identical). CPU smoke:
3-study cache, 1 epoch, `--snapshot-epochs 1` → `snapshot …/encoder_ep01.safetensors`, verify
`ok` + `t1_ok`, 175 tensors; `tests/test_ssl_pretrain.py` 4 passed. REVISE (not done): the kernel
should print the constants it inherits (R2), the `ssl_epNN` arms need a head-of-queue ablate version
(R3), and the gate is three sessions (~16 GPU h), so it lands after session 4 (R4).

**3. Per-slot aux logits in every OOF npz (0/3 refuted).** `validate()` keeps the model's second
output; `oof_fold_k.npz` gains `aux` [N, S, 12] and `slot_names` (plain strings, no pickle).
`scripts/aux_as_run.py <run> <slot> <out>` writes a pseudo-run whose `logits` = that slot's aux
column, with exactly the keys `compare_arms.py --combine` / `stack_oof.py` read. Smoke: 6-study
cache, 1 epoch → `aux (1, 6, 12)`, helper → `(1, 12)`. This is the prerequisite for reading a
medial slot's own head (item 4b) without any new scoring code.

**4. "medial" zoom centre (applied with corrections from the re-check).** `src/slots.py`:
`ZOOM_CENTERS += "medial"` (coronal bases only, `MEDIAL_BASES`), `MEDIAL_CENTRE_MM = (20 distal,
32 medial)` from the LOCATED joint → an 80 mm window covers rows −20..+60 mm and cols −72 medial..
+8 lateral of the joint; `MEDIAL_FALLBACK_MM = (33, 32)` from the IMAGE centre on locator fallback
(the joint sits at median +13.1 mm below it, so the window lands at ≈ −7..+73 mm — the designer's
+20 was a sign error the re-check caught); unresolved side → distal shift only, counted as
`n_medial_side_fallback`. Names follow `zoom_slot_name` → `COR_FS_Z80M`; legacy names are
bit-identical. The builder persists every study's locator estimate (`train_index.json["joint"]`,
~150 B/study — the free target for a later box-aux arm). Honest restatement (157
sided studies): rows −20..+60 mm reach the 61 mm tibial sMCL insertion in **0/157** (the base crop
does in 33 %); what the slot adds is **magnification (0.357 vs 0.625 mm/px) and medial framing**,
the zoomc mechanism family (+0.0045 / +0.0028 pooled on two seeds), not "the whole sMCL". Tests:
`test_zoom_center_medial_synthetic` (both knees; the band starts at the crop's middle column and
runs off its right edge; the original row assertion needed a rewrite — the harness leg
has 4 background columns) + spec-parsing negatives; `tests/test_slots.py -k zoom` 7 passed.
Kernel `kaggle/cache_builder_zoomm/` (= zoomc with `COR_FS:80:medial`, suffix `zm`, fail-fast on a
code dataset that predates the centre); `kaggle_ops.py push-cachezoomm`. COR_T1 excluded on purpose
(locator falls back 44 % on T1).

**REVISE / DROP (not applied, on purpose).** `zoomm` ablate arm: tuple is ready
(`("zoomm", ["--epochs", "8"], "slots_P224_g10t1_zm")`, budget 110–115 min incl. the 14.4 GiB copy)
but session 4 already sums to ≈ 570 min, so it needs a named displacement or the head of session 5;
add `felipedeleon11/slotknee-cache-zoomm` to `kernel_sources` only after the cache is COMPLETE;
gate pooled Δ ≥ +0.0011 with no label < −0.005 (the cor_spec failure mode), MCL read
informational only (8-epoch MCL is the least transferable number); if `flipswap` also adopts,
`COR_FS_Z80M` must be excluded from `_mirror` or the two tested jointly. Baker's silent-negative
repair: the 2,123 A-silent cells are supervised by distillation (teacher mean 0.262 there), so an
arm must transform label AND teacher on the same mask — lowest priority (+0.001–0.003). Box-aux
localisation: later session, targets now exist in the zm index. Dropped: SK_DEC_DEPTH env knob
(env vars never reach a UI run), "one LB submission decides" (POLICY 1), any synovitis re-fill arm,
SSL stage 2 inside this proposal.

**Build-quality pre-gate for zoomm before any GPU time**: read only `cache_builder_report.json` /
`train_index.json` (never the payload): COR_FS joint fallback ≤ 20 % and `n_medial_side_fallback`
≤ 3 % of studies. **Local dry run (80 laptop studies, G=10, T=1, 2 workers, 0.28 GB): joint
fallback 12/80 = 15.0 %, side fallback 2/80 = 2.5 %, located joint median +13.9 mm below the
image centre (design measurement 13.1), medial dx sign = the side split (35 L / 45 R+unresolved);
three rendered slots show the joint line in the upper third and the medial compartment magnified
(`scratchpad/apply/medial_local/medial_3.png`).** Both pre-gates pass on real data.

**Shipped 09-08 08:25**: full suite **493 passed** (492 + the medial test) → `package` →
`push-code` = `slotknee-code` **v33** (status `ready`; carries the medial centre, `aux_as_run.py`,
OOF `aux`, SSL defaults, teacher v3) → `push-cachezoomm` = `slotknee-cache-zoomm` **v1 RUNNING**
(CPU). **COMPLETE the same morning: 37.0 min build, 4,407/4,407 studies, 0 decode failures,
14.43 GB, S = 7, test path 3/3.** Report (`cache_builder_report.json`, fetched via the session-output
listing; never the payload): COR_FS joint fallback **660 / 4,248 = 15.5 %** (gate ≤ 20 %), unresolved
side **72 / 4,407 = 1.63 %** (gate ≤ 3 %), sides L 2,038 / R 2,297; `COR_FS_Z80M` missing only where
COR_FS is missing (159). Both pre-gates pass, so `felipedeleon11/slotknee-cache-zoomm` may now be
added to the ablate kernel's `kernel_sources` — done in **v28** (see below).

### 2026-09-08 (afternoon) — zoomm wired in, SSL kernel baked, and a real bug the review found

**Kernels pushed** (API; both pre-flights stop at the P100 guard as designed): `slotknee-ablate-g10`
**v28** = v27 + `slotknee-cache-zoomm` attached + the `zoomm` arm at position 6 + a refreshed header
(attached sources now match `kernel-metadata.json`; 8-epoch arm ≈ 95–100 min, not 2.5 h);
`slotknee-ssl` **v6** = the S5 constants passed explicitly (`--blr 1e-4 --layer-decay 0.8 --mask-ratio
0.75 --snapshot-epochs 10 20 30`, recorded numerically in `ssl_report.json` with the snapshot files)
plus a fail-fast when the attached code dataset lacks those flags. An independent re-check
refuted nothing, and an end-to-end run trained a micro model on the local 7-slot medial cache and
ran `infer_slotknee.py` on raw DICOM with it: the checkpoint's `slot_layout` carries the 7 names and
`zoom_spec [["COR_FS", 80.0, "medial"]]`, inference rebuilds the medial slot from DICOM, the OOF npz
carries `aux (N, 7, 12)`.

**Placement fact (not a mechanism)**: the ablate kernel has no skip-already-done logic; every run
walks `ARMS` from the top in a fresh `/kaggle/working`. Session 4 (v28) runs
`flipswap → auxslot → central → lora_r8` (≈475 of 500 min; `slotemb` starts only if `lora_r8` finishes
under ~155 min and is then truncated). Session 5 is a **hand re-cut**: drop the finished arms, lead with
`slotemb → zoomm`, set `SK_RESCORE_ARM` to an arm that exists, push v29.

**Bug found and fixed (found twice, independently)**: the three mirror helpers —
`SlotKneeDataset._mirror` (`--flip-swap`), `infer_slotknee._mirror_study` (`--flip-tta`) and
`rescore_oof.mirror_study` — decided by the slot-name *plane prefix*, so a medial-centred zoom slot
(`COR_FS_Z80M`) was column-flipped like a base coronal slot while the label swap read its output as
LATERAL. A mirrored medial crop is still medial anatomy (measured on the 3 test studies: mean |diff|
51–69 grey levels vs the true lateral-anchored crop). Fix: `src.slots.is_medial_slot()` (suffix `M`
from `zoom_slot_name`) and, on every mirrored pass, that slot is **zeroed and its mask cleared** in all
three helpers (`_mirror_mask`, `mirror_mask`); `--flip-tta` prints which slots it drops, and its help
text no longer claims exactness for medial slots. `tests/test_flip_swap.py` pins all three. Consequence
for the queue: `flipswap` × `zoomm` is now safe to stack (the mirrored sample simply lacks the medial
slot, which the model already tolerates via slot dropout) instead of being label-inconsistent.

**Other review notes, recorded not fixed**: the zm `train_index.json` still says `zoom_center =
"image"` next to `zoom_spec … "medial"` (the per-entry spec is what names the slot; only a reader of the
legacy field would be misled); `ssl_report.json`'s "v6" numbering rests on this runbook (the API does
not expose version numbers); `infer_slotknee.py` prints the checkpoint layout, not the rebuilt tensor's
slot count.

**Git / GitHub**: everything up to the design-review items is committed locally as `181add6`
(168 files; the pre-commit `src.train` smoke test passed). The GitHub repo is
`Pipe10101/rsna-knee-detection.` (trailing dot), **public and empty**; `origin` in this checkout points
at the wrong name; `gh` is not installed and there is no Homebrew. Pushing and opening the PR are
Felipe's (commands in the session report). Kaggle's code-sharing rule applies to a public repo during
the competition — decide before pushing.

### 2026-09-09 — SESSION 4 MEASURED (v28, T4, 466 min): central ADOPT, flipswap marginal, auxslot NULL, LoRA REGRESS

Felipe launched v28 on a T4 the evening of 09-08; all four planned arms completed on folds 0-1 at 8
epochs (5.5 min/epoch, 44 min per fold; LoRA 10.5 min/epoch), plus the post-session mirror-TTA rescore.
Gate: `compare_arms.py` against BOTH 8-epoch controls — `recipe_base_s42` (session 1, fetched from
kernel version 13: **0.8650** pooled on these folds) and `recipe_base_s1337` (session 2, **0.8662**).
Their difference, 0.0012 pooled, is the seed noise for 8-epoch arms on this layout, so the gate is
**0.0023**. The 0.0005 floor quoted since session 2 rested on s42 = 0.8657, a number the OOF on disk does not
reproduce (0.8650 under every scoring rule; the origin of 0.8657 is unresolved — a transcription error or a
different run); an independent recompute of every table entry below matched to four decimals, and a paired
bootstrap over the 1,760 studies (400 resamples, vs s1337) gives central **[+0.0003, +0.0051]** (P(Δ≤0) = 0.013),
flipswap **[−0.0008, +0.0036]** (P(Δ≤0) = 0.095), auxslot [−0.0016, +0.0021], lora_r8 [−0.0226, −0.0167].

| arm | Δ vs s42 | Δ vs s1337 | focus Δ (ACL/MCL/LatMen) | verdict | per-label (vs s42) |
|---|---|---|---|---|---|
| `central` (t35 cache: 10 anchors over the central 30 %) | **+0.0038** | **+0.0027** | +0.012 | **ADOPT on both controls** | ACL +0.015, LatMen +0.013, Baker's +0.011, MCL +0.008, MedOA +0.005; Synovitis −0.004, MedMen −0.003 |
| `flipswap` (mirror aug + medial/lateral label swap, p 0.5) | +0.0027 | +0.0015 | +0.016 | **MARGINAL** (adopt vs s42, null vs s1337) | LatMen **+0.053**; MedMen **−0.015**, MCL −0.008 |
| `auxslot` (slot-identity aux head 0.1) | +0.0014 | +0.0002 | +0.004 | NULL | Baker's +0.010, ACL +0.007; Contusion −0.004 |
| `lora_r8` (LoRA r 8 on all blocks + patch embed, lr 1e-4) | **−0.0183** | **−0.0195** | −0.014 | **REGRESS** | every label down; MedMen and Baker's −0.06 |
| mirror TTA on the flipswap model (rescore step) | −0.0003 / +0.0002 per fold | | | NULL | LatMen −0.006 / +0.001 |

Reading: (1) the forum's "contiguous central slices" transfers (+0.018 there, +0.003–0.004 here on a
layout that already covers the stack) and is the first input-side adopt since zoomc; (2) the mirror
augmentation is a **redistribution** — it pools medial and lateral meniscus data, lifting the weak
LatMen column by 0.05 at the cost of MedMen; the net is inside the control-to-control spread, and TTA
adds nothing; its real test is at 16 epochs on top of central (session 5). Design caveat: the swap is
exact only for the four paired labels — MCL and Baker's are medial-only and stay unswapped on the
mirrored view, a mild inconsistency that matches their −0.005/−0.008 here; (3) the research's top
lever, LoRA on all blocks, is refuted at this setting — 8 epochs at LoRA lr 1e-4 is 0.02 behind
partial fine-tuning on every label. It may be under-trained (LoRA typically wants 30–60 epochs) but a
try costs a full session; parked; (4) the slot-identity aux head does nothing on top of the existing
aux slot loss.

**Two things the adoption exposed, both fixed today** (code dataset re-push pending the tests):
- `infer_slotknee.py` rebuilt every study with the default anchor band (trim 0.15) because the
  checkpoint's `slot_layout` never recorded the training cache's `trim_frac`. A t35-trained model
  would have been read with the wrong slices at test time — a silent train/test mismatch that would
  have erased the central gain on the LB. Now `slot_layout` carries `trim_frac` and `crop_mm`
  (from the cache index) and `_decode_study` honours them; `tests/test_layout_trim.py` builds a
  t35 cache, trains one epoch, and checks the rebuilt tensor equals the 0.35 build and differs
  from the 0.15 one. Older checkpoints default to 0.15 / 140 mm (what they were trained on).
- the ablate kernel's rescore step hard-coded the g10t1 cache; it now uses the arm's own cache.

**Session 5 (kernel v30 — v29 was the quota-refused push; re-cut; ~455 of 500 min at the measured 88 min per 8-epoch arm)**:
`central_e16` (16 ep, t35 cache; gated vs `recipe_e16_s42`, folds 0-1, noise 0.0005: does the adopt
transfer?) → `central_flip_e16` (same + `--flip-swap 0.5`; gated vs `central_e16`: flipswap's own
contribution at the real budget) → `slotemb` (8 ep). Budget check (simulated): at the measured
5.53 min/epoch the three arms end at ~457 min, no fourth arm starts (< 60 rule) and the rescore runs;
at 5.9 min/epoch `slotemb` still completes with ~3 min of slack; at 6.5 min/epoch `slotemb` is truncated
(harmless: it is the least important arm and the kernel now flags truncated arms). Whichever of the first two wins is the
candidate accuracy recipe → retrain the 5-fold on the t35 cache (`slotknee-train-final`, ~7 h),
re-push code first (trim-aware inference), then `slotknee-submit`. `zoomm` heads session 6. Per-slot
aux logits are in every session-4 OOF (`aux` (N, 6, 12)); the medial (b) read waits for `zoomm`.

**Also landed 09-09 (after the re-check of the above)**:
- `infer_slotknee.py` refuses a `--ckpt` set whose members do not share one input layout
  (`_check_layouts`: G, T, slot names, zoom, `trim_frac`, `crop_mm`); a t35 fold mixed with 0.15 folds
  or the student would otherwise be decoded on the first member's band. Both submit kernels now include
  `trim_frac`/`crop_mm` in their layout-grouping key (edited, not pushed: pushing a submit kernel starts
  an inference run). `tests/test_layout_trim.py` covers both.
- the ablate kernel records `<arm>/status` (epochs requested vs done per fold, `truncated`) and prints
  `** TRUNCATED -- not comparable **` when `--limit-minutes` cut an arm short.
- code dataset re-pushed twice (trim-aware inference, then the layout guard); the API listed v36 after
  the first.
- **GPU quota is exhausted for this week** (`push-ablate-g10` → "Maximum weekly GPU quota of 30.00 hours
  reached"; session 4 was 7.8 h). v30 was pushed with the GPU flag off so the code is on Kaggle; launch
  it from the UI (T4 x2) after the weekly reset. The submit (accuracy) and student runs also wait for it.
- the runbook's 0.8657 / 0.0005 history is annotated in place (sessions 1 and 2); CURRENT STATE now
  carries the measured 0.0012 / gate 0.0023.

### 2026-09-10 — pushed to GitHub; quota still exhausted; the per-slot read is in

- **GitHub**: `main` and the working branch are on `Pipe10101/rsna-knee-detection.`
  (public; three commits `181add6` → `3fa75ac` → `591a309` + this docs commit). The PR is opened from the
  compare URL (no `gh` on this Mac). `origin` in the checkout still needs `git remote set-url origin
  git@github.com:Pipe10101/rsna-knee-detection..git`.
- **Kaggle**: the weekly GPU quota is still exhausted (a GPU push of ablate-g10 is refused). Session 5 is
  kernel v30 (the latest saved code); launch it from the UI with T4 x2 after the reset, then
  `slotknee-submit` (accuracy) and the v3-teacher student.
- **Per-slot aux read** (docs/research_architecture_20260907.md, 2026-09-10): the fused head beats every
  single slot's own head on all 12 labels (macro 0.869 vs 0.825 for the best slot, AX_FS); winners are
  anatomical (COR_FS menisci/MCL/OA, AX_FS Baker's/effusion, SAG_FS synovitis); stacking a slot head onto
  the main head regresses (−0.011 / −0.015) → **closed**. The medial slot's head is an informational read
  only; `zoomm` is judged by the pooled OOF like every other arm.

### 2026-09-15 — the SSL encoder exists (v6 ran on 09-08), quota is back, session 5 re-cut around it

**`slotknee-ssl` v6 COMPLETE** (Felipe's UI launch 09-08 16:27, concurrent with session 4): 213,340 slices
(the g10t1 cache, absent slot-groups skipped), 1,667 steps/epoch at bs 128, **593 img/s** on the T4 (the
233 img/s pre-flight estimate was 2.5× too pessimistic), all **30 epochs in 190 min**, base lr 5e-5, mask
0.75, layer decay 0.8. MAE loss 0.817 → 0.435, flattening (last five epochs 0.4371 → 0.4357). Outputs:
`encoder_ep10/20/30.safetensors` + `encoder.safetensors` (82.5 MB, 175 tensors, strict load and T=1 load
verified in-kernel; re-verified locally today). 6.4 M samples seen, below S5's 8–10 M saturation point,
so a longer stage 1 is possible later; the gate comes first. Two caveats the re-check recorded: (a) the
loss plateau at epochs 25–29 coincides with the cosine LR reaching ~0, so it is not evidence of
convergence; (b) the MAE saw T=1 slices replicated to three channels with per-channel ImageNet stats,
while the downstream T=1 model uses the RGB-summed patch kernel with averaged stats — not the same linear
map (final-token relative difference ≈ 0.23, cosine 0.97 on real slices; plain DINOv2 has the larger
mismatch, 0.33). The gate measures the net effect, so neither changes the plan.

**v32 kernel hardening (re-checked, not refuted)**: the loop now stops before an arm whose
`EXPECTED_MIN` exceeds the time left (16-epoch arms 185, 8-epoch 95, zoomm 110) instead of the flat
60-minute rule that could start `central_flip_e16` truncated on a fast T4; the `@ssl:` resolver checks the
snapshot's own metadata (backbone reg4, `epoch` = the number in the filename, `ssl = mae`) so a same-named
file from any other source is refused; the rescore step evicts the loop's local cache before copying its
own (one ~12 GB cache in `/kaggle/temp` at a time).

**Quota** reset between 09-10 and 09-15 (a GPU push of ablate-g10 succeeded → v31, a P100 pre-flight
that also restored the kernel's GPU setting). Nothing else ran in between: the 16-epoch accuracy entry is
**still unsubmitted** (last submission 08-23, 0.866).

**Session 5 re-cut (v32) — information per GPU hour**: `central_e16` (16 ep, t35; the confirm the 5-fold
retrain needs) → `ssl_ep30` → `ssl_ep10` (8 ep each on g10t1, `--pretrained-path @ssl:encoder_epNN`
resolved to the attached `slotknee-ssl` output; gate vs both 8-epoch controls, 0.0023, MCL / LatMen not
down) → `slotemb` (starts with ~130 min left at the measured rates). Session 6: `central_flip_e16` (vs
`central_e16`) → `ssl_ep20` (only if ep10 and ep30 disagree) → `zoomm`. The SSL decision has to precede the
7 h 5-fold retrain (it changes the encoder init), which is why it moved ahead of `central_flip_e16`.
The `@ssl:` resolver skips an arm whose file is not found, so an `ssl_*` name can never silently train
from the DINOv2 init.

**Observed 09-15**: the GPU-enabled API push of v32 did not fail at the P100 guard — the session kept RUNNING
past 17 min, i.e. Kaggle assigned a supported GPU to an API push for the first time in this campaign. Session
5 is therefore running from the push (watcher armed; results land in the 2026-09-15/16 section). The
"API = P100" rule is no longer absolute: treat a GPU push as a possible real launch and check status
before pushing a training kernel that must not run yet.

**Launch order this quota week (30 h)**: (session 5 already running, 7.6 h) → `slotknee-submit` on T4 → **Submit** (0.3 h; the LB read of the
16-epoch entry has waited a week) → `slotknee-ablate-g10` v32 session 5 (7.6 h) → `slotknee-train-student`
v5 (1.6 h) → session 6 (6.3 h) → the 5-fold retrain on the winning recipe (7 h). Total ≈ 23 h.

### 2026-09-15 (evening) — SESSION 5 MEASURED (v32, 472 min): central confirmed at 16 epochs, SSL encoder REGRESSES, slotemb null

The v32 push ran to completion on its own GPU (no P100 guard hit). Four arms, all complete, none
truncated (the new per-arm budget rule stopped the loop before `central_flip_e16` with ~28 min left).

| arm | control | Δ pooled | focus Δ | verdict | per-label |
|---|---|---|---|---|---|
| `central_e16` (16 ep, t35 central-block cache) | `recipe_e16_s42` 0.8822 (folds 0-1) | **+0.0025** (0.8847) | +0.0153 | **ADOPT** (gate 0.0023) | LatMen **+0.028**, ACL **+0.015**, MCL +0.004, MedOA +0.004, Fracture +0.004; MedMen −0.011, Effusion −0.004, Synovitis −0.004 |
| `ssl_ep30` (8 ep, DINOv2 → MAE-continued encoder, epoch 30) | `recipe_base_s42` 0.8650 | **−0.0779** (0.7871) | −0.060 | **REGRESS** | every label down; Baker's −0.156, Contusion −0.146, Fracture −0.129 |
| `ssl_ep10` (same, epoch 10) | same | **−0.0746** (0.7905) | −0.056 | **REGRESS** | same pattern; ep10 ≈ ep30 |
| `slotemb` (8 ep, per-slot token embedding) | same | +0.0003 (0.8653) | −0.0005 | NULL | nothing beyond noise |

Reading: (1) **central transfers to 16 epochs** — the third measurement of the same lever in the same
direction (+0.0038 / +0.0027 at 8 ep, +0.0025 at 16 ep; ACL and LatMen up every time). The 5-fold
accuracy model is therefore retrained on the t35 cache at 16 epochs; expected pooled ≈ 0.884 vs 0.8812.
(2) **SSL stage 1 is CLOSED** for this recipe: with 8 of 12 blocks frozen, the MAE-continued encoder is
0.075 worse than the DINOv2 init on every label, and epoch 10 equals epoch 30, so it is not "too little
pretraining" — MAE features need full fine-tuning, and full fine-tuning (tb12, lrb1e4, LoRA) is refuted
on this dataset. A frozen-feature SSL (DINO-style, not MAE) would be the only remaining variant; not
queued. (3) `slotemb` joins `auxslot`: slot identity brings nothing on top of the aux slot loss.

**Session 6 (v33)**: `central_flip_e16` (16 ep, t35, `--flip-swap 0.5`, vs `central_e16` 0.8847 on the
same folds: flipswap's own contribution at the real budget) → `zoomm` (8 ep) → `recipe_e24` if time. The
`ssl_ep20` arm is dropped. Quota this week: session 5 used 7.9 h; the 5-fold retrain (7 h), the accuracy
submit (0.3 h), the student (1.6 h) and session 6 (5 h) fit the remaining 22 h.

**Retrain**: `slotknee-train-final` on `slots_P224_t35_g10t1`, 16 epochs, otherwise the adopted recipe;
its checkpoints carry `trim_frac 0.35` (code dataset 2026-09-09+), and the submit kernel groups by band,
so the new five folds must replace the old ones in `slotknee-models` (`push-models` with the new dir +
the student dir) before `slotknee-submit` runs.

### 2026-09-16 — session 6 and the retrain: the accuracy entry becomes a two-band ensemble

Both overnight runs completed from their pushes (`slotknee-train-final` v13 and `slotknee-ablate-g10` v33
started on Kaggle-assigned GPUs).

**Session 6 (v33, folds 0-1)**

| arm | control | Δ pooled | verdict | per-label |
|---|---|---|---|---|
| `central_flip_e16` (16 ep, t35, `--flip-swap 0.5`) | `central_e16` 0.8847 | **−0.0002** (0.8845) | NULL → **flip-swap CLOSED** | LatMen +0.018 but MCL −0.013, ACL −0.004, Fracture −0.005; mirror TTA on it +0.0007 / +0.0008 (null) |
| `zoomm` (8 ep, + `COR_FS_Z80M` medial slot) | `recipe_base_s42` 0.8650 / `s1337` 0.8662 | +0.0028 / +0.0016 | MARGINAL (zoomc-class) | MedMen +0.012, ACL +0.006, MCL +0.004, no label below −0.003 |

The mirror augmentation keeps redistributing (LatMen up, MCL and ACL down) and nets nothing at the real
budget — closed. The medial slot behaves exactly like the joint-line zoom did (+0.003–0.005, control-
dependent): adopt-for-accuracy-only material, and only after a 16-epoch stack on the t35 cache; parked.

**The retrain (`slotknee-train-final` v13, t35 cache, 16 ep, 5 folds, 4,407 studies)**: pooled OOF
**0.8836** vs 0.8812 for the coverage folds — +0.0023, the fourth positive measurement of the central
lever (ACL +0.013, MCL +0.009, LatMen +0.022; MedMen −0.011). On the 58 expert labels the two sets tie
(macro over 12 labels 0.8806 vs 0.8805; MCL 0.927 vs 0.887, ACL 0.945 vs 0.931, LatMen 0.717 vs 0.758).

**Ensemble test (`compare_arms.py --combine`, all 4,407 studies)**: rank-averaging the two 5-fold sets
gives **0.8843** — +0.0031 vs the coverage folds and +0.0008 vs the central folds alone, every focus
label up. POLICY 1 selects by pooled OOF, so the accuracy entry is **all ten checkpoints in two layout
groups** (the submit kernel groups by band since 09-09 and rank-averages the groups per finding).
`slotknee-models` now holds `fold_0..4` (central band), `g10t1_fold_0..4` (coverage band) and the
student; the accuracy kernel's default glob is `*fold_*_best.pt` with `student_*` excluded.

**Expected LB** for this entry: ~0.895 by the OOF→LB offset seen so far (+0.013), higher if the forum's
"LB > OOF" pattern holds. Efficiency entry unchanged (the student, v2 teacher).

### 2026-09-17 — both entries scored; per-label band weights shipped; a full red-team review; and a directory wipe

**Leaderboard.** The accuracy entry (two 16-epoch 5-fold bands, rank-averaged, TTA) scored **0.905**
public; the efficiency entry (the single full-fit distilled student, TTA off) scored **0.907** the same
day. They are inside the public split's noise (±0.005 on ~390 studies), but the direction matters: one
model trained on all 4,407 studies ties ten fold models. Full-fit cannot be OOF-gated, so this stays an
observation, not a rule — and the review below makes pre-registering the selection rule a priority.

**Per-label band weights.** The 5-fold OOFs were re-stacked per label with nested left-one-fold-out
validation: weights over (central, coverage) of ACL [0.95, 0.05], MCL [0.8, 0.2], Medial Meniscus
[0.05, 0.95], Lateral Meniscus [1.0, 0.0], the rest near equal; +0.0010 pooled [+0.0005, +0.0015] vs
equal weights, transferring +0.0010 twice to independent folds-0-1 models. Shipped as
`stack_weights.json` in `slotknee-models` (the submit kernel globs `stack_weights*.json` and applies it
with no code change; the log line `stack weights: … prefixes=['', 'g10t1']` confirms it loaded) and
submitted as `slotknee-submit` v21. **The review flags this as a live risk**: +0.0010 is below the
0.0023 gate, the weights are extreme on one label, and the file auto-applies to *any* future run of the
kernel, including a re-score of an older submission. Rename it to an explicit `_ADOPTED` name, or pull
it, until the LB read decides.

**Red-team review — `docs/red_team_20260916.md`.** A systematic attempt to break the project along
eight dimensions (validation, train/serve skew, labels, code, Kaggle operations, rules and licences,
experimental design, documentation): **65 findings, 60 confirmed, 5 partial, 0 refuted** (4 critical,
18 high after re-grading). Nothing found invalidates the two scored entries — the pixel path is
bit-identical between cache and inference, the deployed checkpoints are byte-identical to the local
ones, the fold split reproduces exactly, best epoch equals last epoch on every fold (so the OOFs carry
no selection optimism), and the four large adopts are 7–14× any plausible noise. What it did establish:
the gate cannot resolve what it has been asked to resolve (a null arm passes 11–15 % of the time; the
expected best of 40 null arms is +0.0041, and every adopt since 09-06 sits inside that band); scanner
grouping covers only 624 of 4,407 studies, the rest being report-hash singletons; the v4 label blend
points ~1,750 cells the wrong way where reader A is confident and reader B is silent; and final
selection and the release path are not pre-registered. The prioritised fix plan is section 4 of that
document.

**Directory wipe and recovery (2026-09-17, 04:33–06:23).** The project directory was reduced to a
pre-pivot August snapshot: `.git`, `docs/`, `kaggle/`, `data_subset/` and every SlotKnee module were
gone, and `src/` held the old Phase-1 pipeline. Recovered in place: all 153 tracked files from the
GitHub remote at `b280f22`; the label files (`llm_labels_*`, `teacher_g10_*`, `train_gold.csv`,
`fold_image_uids.txt`) from the `slotknee-code` dataset archive, which is the only copy of the
gitignored `data_subset/labels_external`; and the competition CSVs from the competition API. A copy of
`data_subset` and a full `git bundle` now live outside the Desktop. **Lost for good**: the uncommitted
working tree (the unmeasured trainer changes, and the local `contrib/` and `backup/` branches) — which
also removed the review's critical #1, since the restored trainer selects the best epoch on
`val_auc_derived`, not the 12-study gold subset. **Still missing**: the 649 local DICOM studies under
`data_subset/train_images`; they are only needed for local cache builds, and `fold_image_uids.txt`
records exactly which uids they are, so the fold split is unaffected. A second process was observed
writing into the directory during the recovery (recreating pre-pivot files at 06:19–06:22), so the tree
is contested: check `git status` before trusting it.

