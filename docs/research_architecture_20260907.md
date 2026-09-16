# Architecture research, 2026-09-07 — feature engineering and modeling levers, ranked against what this project has measured

> **Executive summary (after a six-part literature review; details in §S1–S6 below).**
> 1. **Submit before optimising.** Top teams on this competition's forum report LB *above* OOF
>    (CV 0.87–0.90 against report labels → LB ~0.94). Our 0.88 OOF is scored the same way; the
>    "0.05 gap" is partly a metric artefact and only a submission sizes it.
> 2. **The encoder is adapted in the wrong regime.** Fine-tuning the last 4 blocks is the worst
>    option in every published head-to-head; LoRA across all 12 blocks + patch embedding is the
>    best-evidenced (AnyMC3D +0.11 vs frozen; LoRA ≥ full FT below ~10 k studies). This also
>    explains the collapsed attention head. Our closed full-FT arms ran 5–50× too hot.
> 3. **Three cheap, well-evidenced input/label levers we do not use**: horizontal flip with
>    medial↔lateral label swap (aug + TTA; +0.01 aug, up to +0.027 TTA elsewhere); a contiguous
>    central slice block instead of spread anchors (+0.018 on this forum); auxiliary heads on free
>    targets (+0.01–0.03 in three competitions).
> 4. **MCL is the largest model-limited gap, and anatomy explains the failed zoom**: the MCL runs
>    ~95 mm and attaches 61 mm below the joint line; a joint-line crop cuts it. The fix is a tall
>    medial-column ROI as auxiliary localisation, not a hard crop.
> 5. **Closed by the review**: generic label denoising (35 % precision on gold), hedged-label
>    shrink, rotation TTA, metadata tokens, ASL (−0.139 reported). **Decisions for Felipe**:
>    OrthoFoundation-L weights are public (MIT) but trained on gated data; KMAR-50K has a
>    licence conflict (CC BY vs CC BY-NC-ND).

Method: literature and competition write-ups (2023-2026), read against the campaign's own
evidence so that nothing already closed is re-proposed. Every claim is tagged
**[measured here]**, **[literature]** or **[competition write-up]**, with the source.

## 0. The anchor — what is already established here

Recipe: 6 sequence slots × 10 anchors at 224 px / 140 mm, `vit_small_patch14_reg4_dinov2`,
last 4 of 12 blocks fine-tuned (no LoRA, no layer-wise LR decay), CLS+mean per image → 256-d
tokens → 1 mixer layer → per-label attention that has **collapsed to a uniform mean**, aux
per-slot loss, self-distillation 0.5, random 6-of-10 anchor bagging, **16 epochs**. Laterality
normalisation (right knee mirrored) and per-slot 1-99 percentile scaling are already in the
cache builder. Best: **0.8822 pooled on folds 0-1**; the field's single models at the same
backbone family and resolution report **0.929-0.947** [measured here / forum].

Closed by measurement (do not spend on): resolution; attention/MIL/max/top-k/temperature
pooling (5 arms ≤ 0); anchor count; bag 8 vs 6; T=3 triplets (−0.004); per-label distillation;
layout ensembles; multi-bag inference; seed ensembles on the bagged recipe; refined labels;
ViT-B and full fine-tuning of all 12 blocks (`tb12`) on the old layout.

## 1. What the literature says that changes the plan

**1a. The 2D-foundation-model-per-slice paradigm is the right one, and the head is not where
the gain is.** AnyMC3D (Dec 2025) adapts a frozen DINOv2/v3 with LoRA and fuses slice CLS
tokens with a learnable per-task query; it reaches **0.894 mean AUC over 10 3D tasks with 1.3 M
trainable parameters**, beats 3D CNNs at every parameter budget, wins VLM3D, and — the key
number — gains **ΔAUC = 0.11 from adaptation alone** with the same DINOv3 backbone versus
frozen features [literature: arXiv 2512.12887]. The Medical Slice Transformer (Sci Rep 2025)
reaches **0.85 vs 0.69 for a 3D ResNet on MRNet knee meniscus** with DINOv2-S per slice and a
**one-layer transformer with no positional embedding**, and reports that freezing DINOv2 costs
~0.3 AUC [literature: PMC12227771]. A 2025 2.5D-vs-3D study concludes the sequence model
"mattered far less than the paradigm" and that "the stage-1 feature extractor is the most
promising area" [literature: barnacle.ai 2025-11-19]. This agrees with what we measured: five
head variants were null, and "more capacity" arms were null — the head has nothing to select
between because per-slice features are not task-discriminative enough.

**1b. How the encoder is adapted matters more than which encoder.** AnyMC3D: LoRA rank 8,
α 16, on the **patch embedding and all attention projections of every block**, backbone
frozen. The MIDOG-2025 DINOv3 paper: LoRA rank 4 / α 8 / dropout 0.05 on q,v only, ~650 k
params, AdamW 1e-4, wd 0.1, cosine with 10 % warm-up, 60 epochs [literature: arXiv 2508.21041].
A lung-nodule study reports LoRA-DINOv2 **+2.7 % ROC-AUC over standard fine-tuning** with far
fewer parameters [literature: IEEE 10635887]. We fine-tune the last 4 blocks fully and freeze
the first 8 and the patch embedding — the opposite regime. `tb12` (full fine-tuning of all 12)
was null, but that is not the same experiment: LoRA adapts *every* layer including the stem
with a regularised low-rank update, which is exactly the "properly adapted" condition the
ΔAUC=0.11 result refers to.

**1c. Competition winners localise before they classify.** RSNA 2024 lumbar (1st, 2nd): a
keypoint stage predicts anatomical coordinates, crops are sized from anatomy (height = level
spacing), classification runs on 2.5D stacks of the **8 middle frames** through encoder → LSTM →
attention pooling; training uses `0.5·labels + 0.5·pseudo-labels` with **confident-learning
label denoising**, sequence reversal, left/right mixing, manifold mixup, and **nine-rotation
TTA** [competition write-up: github brendanartley/RSNA-2024-Competition]. RSNA 2023 (1st): 3D
segmentation → organ crops → 96 equidistant slices as 32 × 3-channel 2.5D frames → CNN + GRU,
384 px, soft labels weighted by per-slice visibility [competition write-up: github
Nischaydnk/RSNA-2023-1st-place-solution]. Our zoom slots (`zoomc`, `zoomj`) are the *static*
version of this; the persistent MCL gap (0.739 vs labels 0.968) is the finding a learned,
per-study crop would target.

**1d. Missing sequences have a known fix.** MM-DINOv2 (MICCAI 2025): shared patch projection +
per-modality positional embedding + a learnable modality vector added at the **patch** level,
concatenated into one ViT input, with full-modality masking in training. Ablation on glioma
subtyping: concatenation alone MCC 0.40 → per-modality positions 0.63 → modality embedding
**0.74**; external test +11.1 % over supervised SOTA [literature: arXiv 2509.06617]. Our slot
embedding is added *after* the encoder; the encoder itself never knows whether it is looking at
a fat-suppressed or T1 image. AX_T1 is present in 19 % of studies, COR_T1 in 77 %.

**1e. Domain-specific pretraining works even when small.** SB-SSL: slice-based self-supervised
pretraining on **<1,000 knee MRIs**, no external data, ACL AUC **0.954** [literature: arXiv
2208.13923]. BrainDINO: 6.6 M unlabeled slices, "particularly strong advantages under label
scarcity" [literature: arXiv 2604.27277]. The released-weights route is closed: Decipher-MR
(22,594 MRI studies incl. knee, 3D ViT) states its weights "might not be directly shared"
[literature: PMC13230676]; OrthoFoundation has no public release. So SSL on our 264 k slices
(+ KMAR-50K if authorised) is the only way to a knee-aware encoder.

**1f. Vision-language on 4.4 k pairs is marginal.** The MICCAI-2025 recipe (LLM labels →
supervised pretraining → CLIP) is what we already run minus the contrastive stage; MedCLIP's
"10× data efficiency" still means ~20 k pairs [literature: arXiv 2509.13175; MedCLIP]. Low
priority.

**1g. The public solution for this competition** (JunhaoLiXD, DINOv2-S at 224, laterality
mirroring, percentile clipping, 2.5D triplets with gap 2, 6 samples per plane, attention
pooling; public score **0.664**) contains nothing we lack — confirms our preprocessing is
standard and that the gap to 0.94 is not preprocessing [competition write-up: github
JunhaoLiXD/RSNA_Knee_Abnormality_Detection].

## 2. Ranked levers — REVISED after the review (2026-09-07, evening)

| # | lever | kind | evidence (section) | inference cost | est. | cost to build |
|---|---|---|---|---|---|---|
| 0 | **Submit the 16-epoch accuracy entry and read the LB** | measurement | forum: LB > OOF consistently (CV 0.87–0.90 → LB ~0.94) — the "0.05 gap" is partly metric (S2) | — | sizes everything below | one T4 run + a click |
| 1 | **LoRA on all 12 blocks + patch embedding** (r 8, α 16, LoRA lr 1e-4, norms trainable, 30–60 ep) — **MEASURED 2026-09-09 (session 4, 8 ep, lr 1e-4): −0.018 / −0.020 pooled vs the two controls, every label down; REGRESS at this budget.** A 30–60-epoch try costs a full session; parked | modeling | partial last-N FT is the worst regime in every head-to-head; LoRA ≥ full FT below ~10 k; explains the collapsed head (S1) | 0 | +0.01–0.03 | ~120 lines; ~1 session per 2-fold arm |
| 2 | **hflip + medial↔lateral label swap** (aug and TTA) — **MEASURED 2026-09-09/16: +0.0027 / +0.0015 at 8 ep, −0.0002 at 16 ep on the central cache; TTA null both times → CLOSED (a redistribution between the medial and lateral columns, no net gain)** | feature eng. / inference | aneurysm 2nd: +0.01 aug, 8× flip TTA 0.844→0.898 public; lumbar 3rd doubled data this way (S2). Safe under laterality normalisation; MCL/ACL/effusion labels unchanged | ×2 TTA | +0.005–0.015 | ~40 lines (dataset + infer) |
| 3 | **contiguous central slice block** near the joint line instead of 10 spread anchors — **MEASURED 2026-09-09/15: +0.0038 / +0.0027 at 8 ep on both controls and +0.0025 at 16 ep vs recipe_e16 = ADOPT; the 5-fold accuracy model is retrained on this cache** | feature eng. | Kirbiyik +0.018 for 9 adjacent vs 9 spread; crop geometry moved 10/12 labels while encoder scaling moved nothing (S2); Chang 5-slice context > 1-slice (S3) | 0 | +0.005–0.018 | cache builder change (~45 min build) + 1 arm |
| 4 | **slot identity inside the encoder** + masking schedule + full→masked KL | architecture | MM-DINOv2 0.63→0.74 from the sequence vector alone (S4) | 0 | +0.003–0.010 | ~60 lines, 2.3 k params |
| 5 | **auxiliary heads on free targets** (slice position, plane, sequence, PatientSex, field strength) + plain BCE — **slot-identity head (+0.0014 / +0.0002) and in-encoder slot embedding (+0.0003) MEASURED = NULL**; other targets untested | modeling | aux losses were the largest measured lever in three competitions (+0.01–0.03) (S2) | 0 | +0.003–0.010 | ~50 lines |
| 6 | **finding-specific label repair**: synovitis-from-effusion fill — **CLOSED 2026-09-08**: it is already inside `llm_labels_v4_blend.csv` (v2 = reader A + `fill_silent_synovitis`; on our 58 gold rows reader A unfilled 0.678 → v2/v4 0.790 = the forum number; strengthening it, k ≥ 0.75 or syn := eff, is −0.05 on gold and −0.055 on pilkwang's 694 verdicts, and k = 0.5 changes no ranking); Baker's "silent = negative" (2,123 reader-A-silent cells have w = 0 in the base BCE but are DISTILLED toward teacher_g10_oof.csv, whose Baker's on those cells is mean 0.262 / 16 % > 0.5 — an arm must transform label AND teacher on the same cells); hard negatives from OOF FPs | labels | S2; generic confident-learning is closed (35 % precision, S6) but *per-finding semantics* is a different lever | 0 | Baker's arm only: +0.001–0.003 macro (column already 0.964 vs gold) | label + teacher CSV transform |
| 7 | **medial-column ROI as auxiliary localisation** (BB-loss, not a hard crop) + MCL-specific coronal(+axial) head fused late — **the magnified medial slot MEASURED 2026-09-16: +0.0028 / +0.0016 vs the two controls (zoomc-class, marginal); parked; the BB-loss variant untested** | feature eng. / architecture | Tack: BB-loss 0.74→0.94 and beat hard crops; Azcona: late per-plane fusion 0.934 vs 0.858; anatomy of why the joint-line zoom failed (S3) | +1 slot | unknown, targets the largest model-limited gap | new cache slot + head; medium |
| 8 | **OrthoFoundation-L** (DINOv3-L, 894 k knee MRI slices, MIT weights) + LoRA | pretrained encoder | S5 addendum; needs an **eligibility decision** (trained on gated OAI/fastMRI) | ~14× ViT-S, accuracy only | highest ceiling | 2-fold 8-ep gate, one session |
| 9 | SSL (MAE) on own slices, 20–30 ep, low LR, select by OOF; KMAR-50K only after its licence conflict is resolved — **MEASURED 2026-09-15: as the encoder init of the partial-fine-tune recipe, MAE snapshots ep10 / ep30 REGRESS −0.075 / −0.078 on every label; CLOSED for this recipe (MAE features need full FT, which is refuted here)** | pretraining | S5; aneurysm 1st: domain pretraining was "the most important factor" | 0 | +0.005–0.010 | ready (`slotknee-ssl` v5) |
| 10 | full fine-tuning at the *correct* LR (1e-5, LLRD 0.75) | modeling | `tb12`/`lrb1e4` were 5–50× too hot, not refuted (S1) | 0 | ≤ lever 1 | flags exist |
| — | ~~label denoising / shrink~~, ~~rotation TTA~~ (+0.001 reported), ~~metadata tokens~~, ~~sequence heads~~ (contested), ~~ASL~~ (−0.139 on the forum) | | closed or deprioritised by the review | | | |

## 3. Why 1 is first

Every measured null on the head (attention, MIL, max, top-k, temperature, mixer) and on
capacity (ViT-B, tb12, lr) fits one story the literature now supports: with 8 of 12 blocks and
the patch embedding frozen, per-slice features are generic natural-image features, so a query
head has nothing task-specific to select — it degrades to the mean, and more head capacity or
more encoder capacity (full FT of the same 4 blocks, or all 12 without regularisation) cannot
fix that. LoRA across the whole encoder is the regime in which AnyMC3D's query pooling *does*
work and MST's fine-tuned DINOv2 beats 3D by 0.16 on knee. It costs nothing at inference, is
~1.3 M parameters, and is one arm.

## 4. Session-4 queue — DEPLOYED 2026-09-07 (`slotknee-ablate-g10` v24, verified by pull-back)

    flipswap (8 ep)  →  auxslot (8 ep)  →  central (8 ep; cache building)  →  lora_r8 (8 ep)  →  slotemb (8 ep)  →  recipe_e24  →  sag_spec …

Cheap levers are gated at 8 epochs against the two existing 8-epoch controls (five arms fit);
the post-session step measures mirror TTA on the `flipswap` model. `lora_r8` and `slotemb`
(levers 1 and 4) are implemented and queued (v25); `ortho_L` waits on the eligibility
decision; `synov_fill` is CLOSED — the fill is already in the training file (row 6). **Before any of it: submit.**

**2026-09-08 design review (`docs/slotknee_runbook.md` §2026-09-08)** on rows 6, 7 and 9:
row 6's synovitis fill is closed (above); row 9's S5 recipe is now the `ssl_pretrain.py` default
(blr 1e-4, layer decay 0.8, snapshots 10/20/30, T=1 verify) so `slotknee-ssl` v5 inherits it, and its
gate is three sessions, not one; row 7 is split — the **magnified medial-column slot** exists
(`src/slots.py` centre `"medial"`, `COR_FS_Z80M`, kernel `slotknee-cache-zoomm`, cache
`slots_P224_g10t1_zm`; honest expectation: zoomc-class, +0.003–0.005, because rows −20..+60 mm of the
joint reach the sMCL tibial insertion in 0/157 studies), every OOF npz now carries the per-slot aux
logits (`aux`, `slot_names`; `scripts/aux_as_run.py`) so the slot's own head can be read for free,
and the BB-loss localisation (Tack) is a later session with the zm index's locator estimates as
targets. The `zoomm` arm has no queue position yet: session 4 sums to ≈ 570 min. Anchor correction:
on the 16-epoch model the largest model-limited gap is **LatMen 0.758** (LLM 0.879), not MCL (0.887).

## 5. Explicitly not proposed

More resolution, more anchors, bigger backbones, head variants, layout ensembles, T=3, hflip
augmentation or TTA (laterality is normalised), external gated datasets (MRNet/OAI/SKM-TEA/
fastMRI are blocked), and any released "MRI foundation model" — none currently ships weights.

## Sources

- AnyMC3D — https://arxiv.org/abs/2512.12887
- Medical Slice Transformer with DINOv2 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/
- 2.5D vs 3D sequence models — https://www.barnacle.ai/blog/2025-11-19-william-2
- DINOv3 LoRA recipe (MIDOG 2025) — https://arxiv.org/abs/2508.21041
- LoRA-DINOv2 lung nodules — https://ieeexplore.ieee.org/document/10635887/
- RSNA 2024 lumbar, 2nd place — https://github.com/brendanartley/RSNA-2024-Competition
- RSNA 2023 abdominal trauma, 1st place — https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- MM-DINOv2 — https://arxiv.org/abs/2509.06617
- SB-SSL knee — https://arxiv.org/abs/2208.13923
- BrainDINO — https://arxiv.org/abs/2604.27277
- Decipher-MR — https://pmc.ncbi.nlm.nih.gov/articles/PMC13230676/
- LLM labels → CLIP (MICCAI 2025) — https://arxiv.org/abs/2509.13175
- 23-condition knee slice transformer (Eur Radiol 2025) — https://link.springer.com/article/10.1007/s00330-025-12052-8
- Public solution for this competition — https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection

---

# Review findings (six research threads, 2026-09-07)

## S6. Label noise — what is supported for report-derived soft labels with a 58-row expert anchor

**Verdict: one cheap, gated, single-pass procedure is worth an arm; most of the popular
noisy-label toolkit is contraindicated here.**

| technique | evidence | applies here? |
|---|---|---|
| OOF-disagreement pruning (confident learning), single pass | PANDA 1st: private 0.940 vs 0.935; RSNA-2024 2nd used it on the last day; retinal cleanlab cycles AUC 0.972→0.979 but **over-cleaning hurts** | **yes**, cell-level, gated on gold |
| BoMD rank-based soft relabel (multi-label CXR, report labels) | NIH→OpenI 89.6 vs DivideMix 72.8 — sample-selection **fails on multi-label** | yes as the relabel rule: target = (1−λ)·given + λ·model |
| co-teaching / DivideMix / CoDis | real-clinical-noise benchmark (arXiv 2512.09315): **below plain CE** | **no** |
| noise-transition matrix / Gold Loss Correction | needs class-conditional noise; report noise is instance-dependent (VisualCheXbert κ 0.31–0.43) | **no** (2×2 per label from 58 rows is unusable) |
| meta-reweighting (L2RW, Meta-Weight-Net) | convention 1 000–2 000 clean rows; unstable below | **no** |
| per-label Platt/isotonic on the 58 | needs ≥100 events per label; AUC-invariant anyway | **no** |
| per-label uncertainty policy (U-Ones/U-Zeros + smoothing) | CheXpert: ±0.01 per label, **direction differs per label** | yes, only where gold gives ≥5 disagreements in one direction |
| report-guided label smoothing (Rep-GLS, hedge phrase → smoothing rate) | MIMIC-CXR 84.1 vs 79.6 BCE | partly already have (soft LLM targets); the *sharpening* of confident phrasing is the untested half |
| further noisy-student rounds | only round 1 reliably helps (87.6→88.1→88.4 then fluctuates) | we already run round 1 — **no** |

**Procedure for this dataset (recommendation, adopted into the plan):**
1. From the 5-fold OOF, flag **cells** (study × label), not studies: per label, t⁺ = mean p over
   given-positives, t⁻ = mean (1−p) over given-negatives; flag a positive if (1−p) ≥ t⁻, a
   negative if p ≥ t⁺. Expect 3–8 % of cells.
2. **Gate on the 58 gold rows first**: precision of the flags = share of flagged gold cells where
   LLM ≠ expert (baseline ≈ 11 %). Proceed only if ≳ 50 %; otherwise the flags are *model*
   errors and pruning would remove exactly the hard positives we need.
3. Soften, don't delete, one pass: flagged target = 0.2·y + 0.8·p (or weight 0.25). Uniform
   per-label distillation was null; this is cell-selective.
4. On gold, count the *direction* of disagreement per label (LLM-positive/expert-negative is
   the likely majority). Apply U-Zeros-style shrink (hedged p ∈ [0.3, 0.7] → toward 0, weight
   0.5) or U-Ones-style, chosen per label only with ≥ 5 gold disagreements; pool the rest.
5. Keep the ×8 gold up-weight; do NOT fit calibration maps on 58 rows.
6. Gate on gold macro AUC (per-label SE 0.05–0.10 → treat < +0.02 macro as noise) **and** on the
   LLM-label OOF not degrading. Realistic ceiling **0 to +0.01**.

Also flagged: check whether gold disagreements cluster on **laterality** (medial vs lateral
meniscus) — a known LLM-labeler failure mode (CheX-GPT; GPT-4o laterality studies).

Sources: github brendanartley/RSNA-2024-Competition · github kentaroy47/Kaggle-PANDA-1st-place-solution ·
nature.com/articles/s41746-024-01424-x · arxiv.org/abs/2203.01937 (BoMD) · arxiv.org/html/2512.09315 ·
arxiv.org/abs/1802.05300 (GLC) · arxiv.org/html/2510.12209 · CheXpert 1901.07031 · Pham 1911.06475 ·
arxiv.org/html/2508.02495 (Rep-GLS) · arxiv.org/abs/2102.11467 (VisualCheXbert) · arxiv.org/abs/1911.04252 ·
arxiv.org/abs/1908.02983 · arxiv.org/abs/2306.05997 · arxiv.org/html/2401.11505v2

### S6 addendum — the gate was evaluated on the 58 expert rows the same day: **FAIL, with a twist**

Cell-level confident-learning flags from the 8-epoch 5-fold OOF: 10.8 % of cells flagged.
On the 58 gold studies (696 cells; LLM ≠ expert in 18.2 %): **flag precision 35 %, recall 22 %**.
Below the 50 % bar → the flags are mostly *model* errors. **The prune/soften arm is closed as
designed** — it would have removed exactly the hard positives the model is worst at.

**The twist: the disagreement has a consistent direction.** LLM-positive / expert-negative
dominates in 10 of 12 labels — Effusion 16 : 0, MCL 7 : 0, Medial OA 7 : 1, Contusion 12 : 3,
Lateral OA 8 : 2, ACL 6 : 1, Baker's 5 : 1. Only Synovitis runs the other way (4 : 14) and its
gold is known-corrupt. No laterality-swap signature (paired medial/lateral labels carry 28 % of
disagreements vs 33 % uniform; 4 both-sides cases). So the LLM labels are not *noisy*, they are
**systematically over-positive relative to expert severity thresholds**: a report's "mild
effusion" or "grade-1 MCL sprain" becomes a soft positive that the annotator does not count.

What survives, and is cheap to gate (label-CSV transform, zero code in the trainer):
- **per-label hedged-positive shrink** (the CheXpert U-Zeros + smoothing analogue, chosen by
  the gold direction): for labels with ≥ 5 gold disagreements in the LLM-positive direction, map
  LLM p ∈ [0.3, 0.7] → p·0.5 (weight 0.5 on those cells); leave confident cells alone. Gate on
  gold macro AUC over 58 (treat < +0.02 as noise) and on the LLM-label OOF not degrading.
  Ceiling per CheXpert's per-label results: ±0.01. Rank-based AUC means this can only help
  by re-ordering *mid-confidence* training targets toward the expert severity bar.

Replaces `labels_cl` in the proposed session-4 queue with `labels_shrink`.

### S6 addendum 2 — the shrink variant is closed too, and the reason closes the whole family

Tested on the 58 expert rows as a *training-target quality* question (does the transformed
label agree better with experts?):

| target | macro-10 vs experts |
|---|---|
| `v4_blend` as is | **0.9089** |
| shrink p ∈ [0.3, 0.7] × 0.5 on the 10 over-positive labels | 0.9055 |
| same, × 0 | 0.8881 |
| shrink p ∈ [0.2, 0.8] × 0.5 | 0.9089 (no cells change) |

No variant improves the target. The mid-confidence band is nearly empty for exactly the labels
that are over-positive (Effusion 1 %, MCL 3 %, Baker's 0 %, Fracture 1 % of studies): **the LLM
disagrees with the experts confidently, not hesitantly**, because the report states the finding
and the expert applies a higher severity bar. A monotone per-label transform cannot fix a
disagreement that lives in confident cells, and rank-based AUC is invariant to it anyway.

**Conclusion for the labels axis**: every cheap label lever is now measured and closed —
re-extraction (reports are the ceiling), refined labels (identical vs gold), uniform and
per-label distillation, confident-learning pruning (35 % precision), hedged shrink (no target
gain). What remains is what the model already does on Effusion / Contusion / Fracture / Medial
OA, where it **beats its own labels** against experts: learning the severity bar from images.
The lever for the remaining five model-limited findings is therefore *vision*, not labels.
`labels_shrink` is removed from the session-4 proposal.

### Feasibility note for lever 1 (checked locally, timm 1.0.28)

`vit_small_patch14_reg4_dinov2` exposes `blocks[i].attn.qkv` (1152×384), `attn.proj` (384×384),
`mlp.fc1` (1536×384), `mlp.fc2` (384×1536) as `nn.Linear` and `patch_embed.proj` as a
`Conv2d(3→384, 14×14)`; `peft` is not installed and is not needed — a ~100-line wrapper
(`W x + (B A) x · α/r`, A∼N(0,σ), B=0) covers it. Parameter budget at r = 8: **0.22 M** for
attention projections on all 12 blocks, **0.59 M** with the MLPs, +7.8 k for the patch
embedding — versus **7.10 M** trainable today (last 4 blocks fully). AnyMC3D's setting (patch
embedding + all attention projections) is the 0.23 M configuration. Zero inference cost; the
merged weights can be folded back into the base matrices for the submit kernel.

## S4. Multi-sequence fusion and missing slots — the encoder never learns which sequence it sees

**Verdict: the single best-supported architectural change is ~2 k parameters, inside the encoder.**

| method | setting | #seq | missing handled | key numbers |
|---|---|---|---|---|
| **MM-DINOv2** (MICCAI'25) | 2,661 glioma pts, ViT-B/14 | 4 | 1-of-4 | ablation MCC int/ext: concat 0.40/0.29 → per-image pos-emb 0.63/0.36 → **+ sequence embedding 0.74/0.49** → + full-sequence masking 0.74/**0.57** |
| MultiMAE brain MRI (MLMI'25) | 3.2 k train, ViT-B | 4 | Dirichlet per-modality masking, absent tokens dropped | missing T1c: MCC 0.47 vs 0.66→−0.10 for plain ViT |
| AnyAD (2025) | BraTS, frozen DINOv2-B | 4 | random modality masking | AUROC 0.947 FLAIR-only vs 0.948 all-4 |
| CCSD (2025) | BraTS seg | 4 | full→subset KL self-distillation | +2–3 Dice over plain modality dropout |
| improved modality dropout (MICCAI'25) | CT + EHR | 2 | **learnable per-modality missing tokens replace zeros** | 0.837→0.840 AUROC; converges < 50 vs > 300 epochs |
| CMPT (2025) | 2 modalities | 2 | learnable proxy tokens + rank-1 LoRA | 75.7 vs 73.5 (dropout) |
| PRA-PoE (2026) | ADNI/OASIS | 4 | availability tokens + **Gaussian product-of-experts** | 63.9 vs mmFormer 51.7 |
| CoPAS (Nat Comms'24) | 1,748 knee MRI, 12 findings | 5 volumes | not handled | learned plane preferences match radiologists: **meniscus → sagittal, MCL/LCL → coronal** |
| TripleMRNet plane study (QIMS'25) | MRNet | 3 planes | — | meniscus: axial best, sagittal worst; ACL: sagittal best, coronal worst |
| Epipolar Transformers / RTF | 2-view | 2 | — | learned cross-view weights *hurt* at ~3 k images/view and help at 300 k; learned fusion overfits the dominant view |

**Implementation facts (from the MM-DINOv2 code).** Every sequence image goes through the
*shared* pretrained patch-embed with the pretrained positional table re-applied from index 0;
tokens are concatenated; `nn.Embedding(n_seq, d)` is added to every patch token of that
sequence. The whole 0.63→0.74 step costs 4×768 parameters. Masking: one random sequence has all
its patches replaced by DINOv2's learnable `mask_token` (student only); test-time "missing" =
the same mask token. Backbone fully fine-tuned after 10 frozen epochs.

**Recommendation for the 6-slot model (d = 384), in order:**
1. **Slot identity inside the encoder**: `nn.Embedding(6, 384)` (2,304 params; or plane(3)+
   contrast(2) = 1,920) added to every patch token after the positional embedding, before
   block 0, **zero-initialised** so step-0 features are unchanged. Keep the post-encoder slot
   embedding. This is the MM-DINOv2 0.63→0.74 step and the best-supported change here.
2. **Missing slots as learnable `[MISSING_s]` tokens** (6×384) with a key-padding mask —
   never zero-fill (two papers: learnable tokens beat zeros).
3. **Masking schedule** matched to the real missing rates (AX_T1 81 %, COR_T1 23 %, … + a
   uniform floor): p = 0.5 mask one present slot, p = 0.15 mask two; add a full→masked KL
   self-distillation term (stop-grad teacher = same network on the unmasked view) — the piece
   that separates CCSD/M3AE from plain dropout. Two forward passes per study.
4. **Late fusion with fixed per-finding plane priors** as the logit-gate initialisation
   (meniscus SAG+COR; ACL SAG; MCL COR; effusion/cartilage all; patellar AX) with a small
   regularised learnable deviation — because learned fusion demonstrably overfits at this data
   scale, and because `cor_spec` already read MCL +0.0215. Equal-evidence alternative: per-slot
   heads fused as a Gaussian product-of-experts (missing slots contribute no precision).
5. Optional per-slot LayerNorm scale/shift (≈ 111 k params) only if (1) plateaus.
6. **No metadata tokens** (TE/TR/fat-sat): no evidence for classification; use them only to
   assign slots.

Uncertainties: MM-DINOv2 is 1-of-4 missing with a fully fine-tuned ViT-B; 1–4 of 6 missing with
a ViT-S under LoRA is extrapolation; no head-to-head of additive slot embedding vs slot-specific
LoRA/norms on same-anatomy grayscale MRI; fixed-prior-beats-learned-gate is inferred from
non-knee evidence plus our own `cor_spec`.

Sources: arxiv.org/abs/2509.06617 (+ github daniel-scholz/mm-dinov2) · arxiv.org/abs/2509.11442 ·
arxiv.org/abs/2512.21264 · arxiv.org/abs/2511.14599 · arxiv.org/abs/2509.18284 · arxiv.org/abs/2501.17823 ·
arxiv.org/abs/2605.13081 · nature.com/articles/s41467-024-51888-4 (CoPAS) · pubmed 40606344 ·
arxiv.org/abs/2410.15847 · arxiv.org/abs/2005.04551 · arxiv.org/abs/2511.03014 · pmc PMC13364302

### S4 addendum — what the current model already does (checked in `src/slotknee.py`)

Absent slots are **not encoded** (`encode_absent=False`); their per-image features are zero, so
their tokens reduce to `slot_embed[s] + group_embed[g]` — i.e. a learned per-slot placeholder
already exists at the token level — and a key-padding mask gives them exactly zero attention.
Slot dropout in training is uniform p = 0.1 over present slots (keeping ≥ 1), with no
full→masked consistency term. So of S4's recommendations: (2) `[MISSING_s]` tokens are
effectively in place; (4) fixed plane priors and (5) per-slot norms are untried; and the two
genuinely absent pieces are **(1) slot identity inside the encoder** and **(3) a masking
schedule matched to the real missing rates plus the full→masked KL**. Those two are what the
`slotemb` arm should carry.

## S5. Domain pretraining at 0.3–0.5 M slices, and the KMAR-50K datasheet

| study | corpus | objective / init | downstream delta |
|---|---|---|---|
| Xiao et al. WACV'23 | 266 k CXR | MAE ViT-S from scratch, **90 % mask**, 800 ep | mAUC 82.3 vs 78.6 (ImageNet-MAE init) vs 79.6 (ImageNet-sup) |
| SB-SSL knee | MRNet ~1.1 k exams | MAE-style ViT-S from scratch, ≤ 70 % corruption | ACL AUC **0.954 vs 0.721** random init; monotone with epochs |
| RAD-DINO | 838 k CXR | DINOv2 objective continued from DINOv2-B (~46 ep) | 197 k-only ≈ full data; largest gains on rare findings; "no-MIM" ablation hurts most |
| low-resource DINOv2 (histopath) | 100 k / 2.5 M tiles | DINOv2 continued; **ViT-S beats ViT-g** | AUROC 0.73→0.89; **saturates at ~8–10 M samples seen** |
| OrthoFoundation | 1.2 M knee X-ray + MRI slices (fastMRI 300 k, OAI 200 k) | DINO continued from DINOv2/v3-L | ACL AUC 97.7 at 50 % labels > DINOv3 at 100 %; PCL +4 |
| SurgeNet | 260 k–4.7 M frames | DINO from ImageNet init | **260 k subset: no gain past epoch 5** |
| BrainDINO | 6.6 M brain slices | DINO + iBOT from scratch | ADNI 0.954 vs DINOv3 0.675; largest at 10 % labels |
| 3D brain MAE vs JEPA | 58.8 k volumes | from scratch | MAE wins classification (0.945 vs 0.854) |
| SparK / low-data SSL comparisons | CT / small | MAE-style vs contrastive | MAE-style more robust to small downstream sets; needs less data than DINO |

**Plan (recommendation, adopted with one caveat):**
1. Keep **MAE** as the objective for a single-T4 budget (reconstruction family has the best
   evidence at this scale for ViT-S; DINOv2/iBOT needs multi-crop teacher–student at bs ≥ 256).
   **Caveat that must be gated**: no paper runs pixel-MAE warm-started from DINOv2 weights, and
   MAE features linear-probe worse than DINOv2's (0.724 vs 0.763 on NIH) — the fresh decoder can
   drag the encoder toward low-level features. Mitigate with a lower base LR (5e-5–1e-4, layer
   decay ~0.8, not the from-scratch 1.5e-4) and **select checkpoints by downstream OOF AUC only**.
2. Mask ratio 0.75 first, then 0.85–0.90 (CXR: 90 % equal or better and 2.5× faster).
3. **20–30 epochs**, checkpoints at 10/20/30 each fine-tuned — saturation is ~8–10 M samples
   seen, and a 260 k corpus saturated by epoch 5 elsewhere; the 800-epoch from-scratch regime
   does not apply.
4. Data: all planes and sequences of the 264 k cache (diversity > count). **No natural-image
   replay** — no study in this set does it; forgetting cost only −1.0 ImageNet top-1 (DINORET).
5. Small structures: no ligament-vs-diffuse ablation exists; MAE is not disadvantaged; knee-
   specific weights helped ACL and meniscus equally (MedNet-FS).
6. Wall-clock estimate (unverified): 4–7 min/epoch on a T4 for 264 k slices at 75 % mask →
   **30 epochs ≈ 2–3.5 h**; read `img/s` from the kernel log after epoch 1 (the runbook's 2 h is
   optimistic if the loader is the bottleneck).

**KMAR-50K datasheet** (Wang, Shi et al., *Sci Data* 2025, doi 10.1038/s41597-025-05439-1):
1,190 patients, 1,444 paired sequences, **62,506 2D images** (≈ 62 k *unique* slices — the
ground-truth volume is a registered rescan of the artifact volume); single centre (Sichuan);
sequences mostly **PD-TSE fat-sat (82 %)**, T1-TSE 227, little T2; planes sag 511 / tra 492 /
cor 441; Siemens 1.5 T 87 % / 3 T 12 %; FOV 170–180 mm, slice 2.5–5 mm; `.nii.gz` per series,
N4-corrected and min-max normalised; no pathology labels. Two Mendeley repos, each a
"Download All" zip (xw7mrg7ntg v6, 95w9f5tzz8 v6); byte size unpublished (~10–25 GB est.).
Conversion is trivial with nibabel (slice → 224 → uint8, same as the cache builder).
**Licence conflict**: the Mendeley/DataCite record says **CC BY 4.0**; the paper text says
**CC BY-NC-ND 4.0** — must be resolved before any prize-eligible use (the competition's rule on
external data is "publicly available, permissive"; ND could be read as forbidding derived
caches). Expected gain from adding it: small (+25 % slices, new scanners; RAD-DINO's 4× more data
gave +0.4 in-distribution).

Uncertainties: MAE-from-DINOv2 is unpublished; KMAR size and licence unconfirmed; OrthoFoundation
cites a GitHub repo (`ytrsk/OrthoFoundation`) — weights availability not verified at the time.

Sources: arxiv.org/abs/2210.12843 · arxiv.org/pdf/2208.13923 · arxiv.org/html/2401.10815v3 ·
arxiv.org/html/2401.04720 · arxiv.org/pdf/2409.17332 · arxiv.org/abs/2601.18250 · arxiv.org/pdf/2501.09436 ·
arxiv.org/abs/2604.27277 · arxiv.org/abs/2509.02379 · arxiv.org/html/2606.13315v1 · arxiv.org/abs/2404.17202 ·
arxiv.org/abs/2308.06534 · arxiv.org/html/2509.06990 · pmc PMC13458554 · doi.org/10.1038/s41597-025-05439-1 ·
data.mendeley.com/datasets/xw7mrg7ntg/6 · data.mendeley.com/datasets/95w9f5tzz8

## S1. Encoder adaptation — the diagnosis is confirmed, and two closed arms were mis-run

| paper | data | regime | result |
|---|---|---|---|
| **AnyMC3D** | 12 tasks, 1.1 k–50 k volumes (CT/MRI incl. RSNA trauma, shoulder MRI) | frozen DINOv2/v3 + LoRA r 8 α 16 on **patch-embed + q,k,v + out-proj, all blocks**; LoRA lr 1e-4, head 1e-3, focal, ≤ 100 ep | frozen + pooling 0.785 → **0.894 (+0.11)**; pooling *under LoRA*: query-attn 0.962, mean 0.958, transformer 0.950, LSTM 0.903; ViT-L > ViT-S by only 0.008–0.029 |
| Medical Slice Transformer | MRNet knee 1,199 pts | DINOv2-S **full FT at lr 1e-6** | frozen collapses 0.94→0.62; knee 0.85; ViT-S ≈ ViT-B |
| Veasey & Amini 2025 | 857 CT nodules | frozen / full FT lr 1e-5 / LoRA r 32 lr 1e-3 / **partial (last layers)** | frozen 0.76–0.81, full 0.84–0.87, **LoRA 0.85–0.89**; partial "far outperformed" |
| Dutt et al. MIDL'24 | 5 datasets 0.6 k–20 k | 17 PEFT methods vs full FT | LoRA 0.88 vs full 0.84 F1; PEFT advantage grows as data shrinks |
| DINOv3 CXR scaling | 78 k–153 k | full FT lr 1e-5 vs LoRA r 16 | at *large* data full FT wins by 1–5 pp — LoRA is a data-regime choice |
| Dynamic-LoRA DINOv3 | 357 images | frozen / **full FT of last 4 blocks** / LoRA | R² 0.72 / **0.70** / 0.73–0.75: unfreezing last-4 fell *below* frozen |
| 3D-neuroimage MIL benchmark | 0.7 k–22 k scans | frozen ViT-B + mean / ABMIL / TransMIL | mean 0.898 ≥ ABMIL 0.894; learned attention localises worse than a fixed positional prior |
| "Less Could Be Better" | CXR | LoRA vs full FT | LoRA wins 13/18 tasks; optimal rank scales with data (8–16 at ~10 %) |

**Why the head collapsed (now explained, not hypothesised).** With an under-adapted encoder,
slice embeddings vary by anatomy and position, not pathology; attention logits have near-zero
variance and the softmax is uniform. Every pooling variant we tried was null for that reason.
Under full-block LoRA, query attention works but is worth only ~0.004 over the mean
(AnyMC3D Table 6) — so keep the per-label query head, expect nothing from it, and use attention
entropy vs log N as the diagnostic after the change.

**Two closed arms were mis-run, not refuted.** `tb12` (all 12 blocks, lr 5e-5) and `lrb1e4` were
5–50× hotter than the full-FT recipes that work (1e-5 to 1e-6). Full fine-tuning at the right LR
is *not* closed; but LoRA is the better-evidenced regime at 4 k studies either way.

**Recipe (adopted):** freeze everything; LoRA **r 8, α 16, dropout 0.05** on q,k,v,
out-proj of all 12 blocks **plus the patch embedding**; train all LayerNorm affines; LoRA lr
1e-4, head + label queries 3e-4–1e-3, norms 1e-4; AdamW wd 0.05 on LoRA/head, 0 on norms; 10 %
warm-up → cosine to 1e-6; clip 1.0; EMA 0.999; drop-path 0.1; **30–60 epochs** with best-val
checkpoint — 16 may under-train LoRA (so the arm costs ~1 session at 2 folds). Keep reg4.
Give W_v a higher LR than W_q if only q,v are adapted. Expected **+0.01 to +0.03**, extrapolated
from partial→LoRA deltas in Veasey and the 357-image study; the CXR paper's warning is that full
FT regains the lead only past ~80 k labelled images.

Sources: arxiv.org/abs/2512.12887 · pmc PMC12227771 · pmc PMC11875634 · arxiv.org/abs/2305.08252 ·
arxiv.org/abs/2510.07191 · pmc PMC13517550 · arxiv.org/abs/2508.21041 · arxiv.org/abs/2604.26807 ·
arxiv.org/abs/2312.02366 · arxiv.org/abs/2401.12215 · arxiv.org/abs/2410.02247 · arxiv.org/abs/2406.10973 ·
arxiv.org/abs/2401.01752 · arxiv.org/abs/2309.16588

## S5 addendum — OrthoFoundation weights ARE released (checked 2026-09-07)

`github.com/ytrsk/OrthoFoundation`: code **MIT**, checkpoint `OrthoFoudation-L.pth`
(no separate weights licence stated), backbone **DINOv3-L**, pretrained on **1,251,655 knee
images — 893,985 knee MRI slices + 357,670 radiographs**, from OAI + fastMRI (~600 k) plus a
private cohort (130,567 patients, PUTH). Reported: ACL AUC 97.7 at 50 % labels > DINOv3 at 100 %.
The earlier research log recorded "no public release"; that is now wrong.

Two things stand between this and an arm, and both are **Felipe's call, not mine**:
1. **Eligibility.** The weights are MIT, but they were trained on OAI and fastMRI, whose data
   agreements are non-commercial and gated — datasets this project has classified as BLOCKED.
   Whether a *derived model* under MIT is "publicly available" in the competition's sense, or
   inherits the data restrictions, is exactly the licence-trap question in the research log
   (§5, 2026-08-24). Prize eligibility may need a host ruling.
2. **Cost.** DINOv3-L is ~14× a ViT-S per image. With LoRA and grad-checkpointing it fits a
   T4 at small batch; a 2-fold, 8-epoch gate (~6.5 h) fits one session and has a fair control
   (`recipe_base_s42` / `s1337` are 8-epoch runs). The efficiency entry would keep ViT-S; this is
   an accuracy-track lever only. AnyMC3D measured ViT-L > ViT-S by only 0.008–0.029 with
   *natural-image* weights; the knee-specific pretraining is the part that could be large.

If eligible, this is the highest-ceiling item on the board: it is the OrthoFoundation result
the campaign wanted to reproduce with SSL, delivered pre-trained on 3× the data.

## S2. Competition winners (RSNA 2022–2025) and THIS competition's forum

| comp / rank | backbone | input | aggregation | label tricks | TTA | notes |
|---|---|---|---|---|---|---|
| Lumbar'24 1st | ConvNeXt-S, EffV2-S (ViTs worse) | keypoint → per-level crop, 5 slices | bi-LSTM → attention-MIL + **aux heads** | ±2-slice jitter | — | aux loss 0.262→0.252 |
| Lumbar'24 2nd (Pan) | maxvit/coatnet | localizer → 64 px crops | plain 2D best | **confident-learning removal (\|y−OOF\| ≥ 0.8) +1 % public+private** | 27 crops/target | |
| Lumbar'24 2nd (Bartley) | 2D enc + LSTM + attn | sagittal only | LSTM→attn | 0.5 label + 0.5 OOF pseudo; seq-flip, **L/R recombination**, manifold mixup | 9 rotations | |
| Lumbar'24 3rd | ResNet18…ConvNeXt-T | CenterNet keypoints | per-slice → attention | **split L/R (hflip axial) doubles data**; aux losses | axial flip | failed: 3D-CNN, 2.5D+attn, LSTM, focal |
| Abdominal'23 1st | CoaT + GRU | seg → organ crop, 96 slices → 32×3ch | GRU, max | slice label = patient label × visibility; **aux seg dice ×0.125 (+0.01–0.03)** | — | |
| Cervical'22 1st | EffV2-S@512 + LSTM | seg → per-vertebra crop, 15 × 5-ch | LSTM | mixup, permutation | — | |
| Aneurysm'25 1st | nnU-Net enc **pretrained on vessel seg (0.794→0.902)** | 140 mm ROI | masked pooling + location transformer | aux seg loss w 1.0 vs cls 0.1 | L/R flip | no aux: 0.876 vs 0.902 |
| Aneurysm'25 2nd | 3D nnU-Net multi-task | ROI 224³ | cross-attn | **hflip + L/R label swap**; hand-corrected labels | **8× flips (0.844→0.871→0.898 public)** | |
| Aneurysm'25 5th/6th/8th | ViT-L/EVA, coatnet, ConvNeXt-B DINOv3 | YOLO crop (+0.03–0.05), 3-slice stacks | max / transformer | hflip+swap (+0.01); relabel 33 negs; **hard negatives from OOF FPs** | — | CNN+RNN end-to-end "poor" |

**This competition's forum (single-model thread and others):** 0.949 Scott Willis (single
fold, small model, ~5 min scoring, local Gemma-4 labels ≈ 0.89 vs gold); 0.943 Arrants (5-fold
ResNet/EffNet @224, 25 min); 0.942 tennogh @288; 0.938 Raptor (64 slices, 140 mm, 336 px; SWA
no gain); 0.936 Yann (single fold ResNet@224 with **encoder pre-training + a longer regularised
schedule**); 0.926 k256 (5-fold @336, LLM labels only).

**Three forum facts that reprice everything:**
1. **LB > OOF, consistently.** Arrants: CV 0.87–0.90 against report labels → LB ~0.94; several
   posters see the same. Our 0.88 pooled OOF is scored against the same report labels — the
   "0.05 gap to the field" is partly a *metric* gap, and only a submission can size it.
   **Submit before chasing OOF.**
2. **Crop geometry and slice position are the drivers, not encoder size.** Kirbiyik: 9
   *adjacent* central slices beat 9 spread over 24 by **+0.018**; stevenleehans: a crop-geometry
   fix moved 10/12 labels (+0.006) while encoder scaling moved nothing; DINOv2-S→B +0.001,
   DINOv3 slightly worse; 224–288 is the sweet spot. All 0.93+ posters crop 140–150 mm — as
   we do — but our 10 anchors are *spread* over the stack.
3. **Label semantics are per finding.** "Not addressed" = absent for Baker's (3 % positive when
   silent) but unknown for synovitis (84 % unaddressed); **filling synovitis from effusion raised
   that column 0.678→0.790** (measured 2026-09-08: that gain is exactly reader A → v2 =
   `fill_silent_synovitis` on our 58 gold rows, 0.6780 → 0.7903; v2/v4 is the file we already
   train on, so nothing is left to harvest — see row 6); soft > hard 3/3 seeds; Kirbiyik measured **Asymmetric Loss −0.139**
   vs weighted BCE; PatientSex is in the test headers (ACL 54 % M vs 32 % F; Medial OA 12 % vs
   45 %); 7 studies are bilateral; scanner-grouped CV is the consensus.

**Five transferable patterns the model lacks** (with cost): (1) contiguous
central slice block near the joint line instead of spread anchors — low; (2) **hflip with
medial↔lateral label swap** as augmentation *and* TTA (safe under laterality normalisation;
MCL stays) — low; (3) **auxiliary heads** on free targets (slice position, plane, sequence,
PatientSex, field strength) with plain BCE — low; (4) finding-specific OOF label repair
(synovitis-from-effusion, hard negatives) — low–medium; (5) encoder domain pretraining — the
aneurysm winner's "most important factor" — medium (GPU).

Sources: the Kaggle write-ups for lumbar 1st/2nd/3rd/4th, abdominal 1st, cervical 1st/3rd/5th,
aneurysm 1st/2nd/3rd/5th/6th/8th/9th (URLs in the review notes); knee forum threads 735304, 733517,
735154, 737597, 733932, 734105, 734004, 738096, 737566, 737696, 736678, 733826, 735826, 735767, 735639.

## S3. Knee-MRI literature — per-finding planes, crops, ceilings, and why the MCL zoom hurt

| finding | best plane / sequence | ROI in the best systems | best reported AUC (system; train size) |
|---|---|---|---|
| MCL | coronal FS, read *with axial* for attachment level; most sensitive signs are fascial oedema and loss of fat demarcation | **none** — full FOV, 4 stacks × 32 slices (Vuskov 2025) | tear **0.95 int / 0.88 ext** (Vuskov; 3,121); non-tear pathology 0.72; "any injury incl. sprain" 0.78–0.80 (CoPAS, 773) |
| medial meniscus | sag + cor PD/T2-FS; roots on sagittal | 3D localiser → per-meniscus subvolume (Rizk); **bounding-box auxiliary loss** (Tack) | 0.93 (Rizk; 8,058); 0.88 (Vuskov) |
| lateral meniscus | same; posterior horn/root hardest | same | 0.84 (Rizk); human sensitivity 78.5 % lateral vs 91 % medial |
| ACL | sagittal oblique T2/PD-FS | YOLO ligament crop → 112 px (Liu); notch patch + 5-slice context (Chang) | 0.965 (MRNet); 0.977 (MPFuseNet); readers 0.99 |
| PF OA | axial + sagittal FS | segmentation → compartment boxes (Astuto) | 0.85/0.80 (Vuskov); cartilage 0.93 (Astuto) |
| effusion / Baker's / contusion | fluid-sensitive, any plane | none | 0.91–0.94 / 0.93 / 0.83 (Vuskov); contusion 0.82→0.70 without T1/T2 branches (CoPAS) |

**Ideas the model does not use, with evidence:** (1) **localisation as auxiliary supervision,
not hard crops** — Tack: full → crop → BB-loss raised medial anterior horn 0.74→0.87→0.94, and
BB-loss beat hard cropping everywhere; removing attention-localisation cost the eClinicalMedicine
system 0.04 at the low end; (2) **per-finding, per-plane heads with late fusion** — Azcona on
MRNet: per-plane ResNet18 + logistic fusion 0.934 vs 0.858 for one multi-plane network;
CoPAS's per-plane baseline beat its own co-plane attention on MCL (0.802 vs 0.782); (3)
**slice context and multi-slice normalisation** — Chang: 5-slice 0.915 vs 3-slice 0.865 vs
1-slice 0.765; ELNet: multi-slice normalisation removed → meniscus 0.904→0.751; (4) a
slice-token transformer over all stacks jointly (Vuskov) reaches MCL-tear 0.95 with no crop.
Preprocessing: histogram/percentile normalisation, no N4; 2D/2.5D ≥ 3D (0.931 vs 0.871);
protocol heterogeneity, not resolution, drives external loss (0.967 → 0.898).

**Why the joint-line zoom hurt MCL (anatomy).** The superficial MCL is ~95 mm long, attaching
3 mm above the epicondyle and 12 mm and 61 mm *below* the joint line; tears are femoral 58 %,
mid-substance 19 %, tibial 23 %, and 60 % of distal grade-3 tears are Stener-like. A joint-line
square crop removes the tibial insertion and the periligamentous oedema that is the most
sensitive sign — the menisci sit at the joint line, the MCL does not. Direction: a **tall,
narrow "medial column" ROI on coronal FS** (epicondyle to ≥ 65 mm distal, including
subcutaneous fat) supplied as *auxiliary localisation* (BB-loss) rather than a hard crop, an
MCL-specific coronal(+axial) head fused late, and a check that the 10 coronal anchors actually
traverse the ligament.

**Ceilings**: ACL 0.89 vs 0.95–0.98 → model-limited; MM 0.83 vs 0.88–0.93 → mostly model-limited;
LM 0.72 vs 0.84 with labels at 0.88 near the human limit → half label-limited; PF OA 0.80 ≈
literature → label-limited (subjective grading); **MCL 0.74 vs 0.88–0.95 → the largest
model-limited gap.**

Sources: MRNet (PLoS Med 2018) · Azcona arxiv 2010.01947 · ELNet (MIDL'20) · MRPyrNet · MPFuseNet
2108.08136 · CoPAS PMC11368947 · eClinicalMedicine 2025 · Vuskov Eur Radiol 2025 (PMC13035746) ·
Astuto 2021 · Liu 2019 · Chang 2019 · Tack 2021 · Rizk 2021 · Fritz 2020 · Germann 2020 · Mead 2025 ·
ESSR recommendations PMC11399221 · LaPrade 2007 · Schweitzer 1995 · MCL tear-location study 2024

## 2026-09-10 — first per-slot read from the aux heads (session-4 OOFs); slot-specialist stacking is closed

Every session-4 OOF carries the per-slot aux-head logits (`aux` (N, 6, 12), saved since 2026-09-08).
Scoring each slot's own head against the labels on the 1,760 held-out studies of folds 0-1 (`central`
arm; `flipswap` and `auxslot` give the same picture) answers "which slot reads which finding" for the
first time from a trained model rather than from the literature:

| label | main head | SAG_FS | COR_FS | AX_FS | SAG_T1 | COR_T1 | AX_T1 | best slot |
|---|---|---|---|---|---|---|---|---|
| ACL | 0.848 | 0.809 | 0.787 | 0.772 | 0.782 | 0.764 | 0.822 | AX_T1 (19 % present) |
| MCL | 0.801 | 0.709 | **0.775** | 0.767 | 0.696 | 0.753 | 0.760 | COR_FS |
| Medial Meniscus | 0.881 | 0.782 | **0.859** | 0.782 | 0.777 | 0.834 | 0.787 | COR_FS |
| Lateral Meniscus | 0.777 | 0.734 | **0.743** | 0.736 | 0.721 | 0.696 | 0.716 | COR_FS |
| Medial OA | 0.913 | 0.870 | **0.895** | 0.880 | 0.877 | 0.892 | 0.883 | COR_FS |
| Lateral OA | 0.855 | 0.824 | 0.824 | 0.827 | 0.822 | 0.809 | 0.842 | AX_T1 |
| PF OA | 0.864 | 0.837 | 0.828 | 0.829 | 0.830 | 0.804 | 0.856 | AX_T1 |
| Effusion | 0.871 | 0.843 | 0.834 | **0.845** | 0.789 | 0.795 | 0.751 | AX_FS |
| Synovitis | 0.868 | **0.851** | 0.828 | 0.842 | 0.821 | 0.758 | 0.784 | SAG_FS |
| Baker's | 0.936 | 0.753 | 0.747 | **0.930** | 0.759 | 0.717 | 0.860 | AX_FS |
| Contusion | 0.902 | 0.835 | **0.864** | 0.843 | 0.731 | 0.755 | 0.699 | COR_FS |
| Fracture | 0.909 | 0.843 | **0.889** | 0.849 | 0.772 | 0.835 | 0.697 | COR_FS |
| macro | **0.869** | 0.808 | 0.823 | 0.825 | 0.781 | 0.784 | 0.788 | |

Facts: (1) the fused head beats the best single slot on **every** label, by 0.006 (Baker's) to 0.039
(contusion); the fusion is worth ≈ +0.04 macro over the best slot; (2) the winners are anatomical —
coronal FS for both menisci, MCL, medial OA, contusion and fracture; axial FS for Baker's and effusion;
sagittal FS for synovitis; the T1 slots are 0.04 weaker across the board; (3) `flipswap` lifts the
coronal slot's own lateral-meniscus read from 0.743 to 0.782, which is the mechanism behind its
+0.05 LatMen column (the mirror doubles the coronal lateral-meniscus examples); (4) the AX_T1 "wins" on
ACL / lateral OA / PF OA are on the 19 % of studies that have an axial T1 and are not comparable.

**Closed**: stacking a slot's own head onto the main head. `scripts/aux_as_run.py` + `compare_arms.py
--combine` (equal-weight rank average) on the central arm: main + COR_FS **−0.015**, main + AX_FS
**−0.011** vs the control (main alone +0.0027). A per-label convex weight (`stack_oof.py`) can only put
≈ 0 on a head that is 0.02–0.05 weaker on every label, so there is nothing to harvest. Consequence for
lever 7 (medial column): the medial slot's own head will be read as information (does a magnified
medial view beat the base coronal slot's 0.775 on MCL / 0.859 on MedMen?), not as a stacking member;
the arm's value is decided by the pooled OOF of the fused model, exactly like zoomc.

