# Pivot research: the strongest *fast* model we can build by Oct 22

*Compiled 2026-08-21 from the live Kaggle competition pages, the discussion forum, the
efficiency leaderboard, recent papers (2024–2026), industry products, and measurements on
this repo's data. Every number below is either quoted from a linked source or measured here.*

---

## 0. The verdict in one paragraph

Build **one small model, not an ensemble**: a DINOv2-Small (ViT-S/14, 22M params)
fine-tuned on **all 4,407 studies using the public LLM-read report labels**, fed
**2.5D triplets of physically-sorted slices from a fixed-millimetre crop**, with a
**per-finding attention head over ~6 sequence "slots"**. That recipe is what the
0.93–0.94 single-model entries on the leaderboard are, it is what the 2025 literature
says beats 3D CNNs, it runs inference on ~1,300 test studies in **under ten minutes on a
T4**, and it is competitive for *both* prize tracks at once. The things that move the score
are the labels, the crop geometry and slice placement — **not** the encoder size (measured:
ViT-S → ViT-B = +0.0011, inside a 0.0020 noise floor). Spend the next nine weeks on the
pipeline and the labels, and keep the network small on purpose.

---

## 1. What the competition actually rewards (read this before designing anything)

Source: [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview),
[data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data),
[efficiency LB notebook](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb).

| Fact | Value | Why it matters |
|---|---|---|
| Metric | macro ROC-AUC over 12 findings | rank statistic; calibration is worthless, rank-average when ensembling |
| Test set | ~1,300 studies; public LB = 30% (~390) | private is ~910 studies; public LB is noisy at the 3rd decimal |
| Reports at test time | **No** — `test.csv` is `StudyInstanceUID` only | text is training signal only; the deployed model reads pixels |
| Runtime limit | ≤ 9 h, CPU or GPU notebook, internet off | pretrained weights must be attached as a Kaggle dataset |
| Deadline | Oct 22 2026 (entry/merge Oct 15) | **62 days** from today |
| Main prizes | $9k / $7k / $6.5k / $6k / $5.5k / 5×$5k | top-10 pay |
| Efficiency prizes | $7k / $6k / $5k | a *second* podium, and a submission can win both |
| Efficiency score | `Eff = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`, **minimise** | see the exchange rate below |

### The exchange rate between accuracy and time

`Benchmark` is the all-0.5 sample submission (AUC 0.5); `maxAUC` is the best private score
(≈ 0.95 today). So the denominator is ≈ −0.45 and

- **0.01 AUC ≈ 720 s ≈ 12 minutes of runtime.**
- An inference run that takes 2 hours costs the same as **−0.10 AUC**.
- A 0.935 model that runs in 10 min *ties* a 0.945 model that runs in 22 min, and *beats*
  a 0.952 model that takes more than ~31 min.

The efficiency leaderboard confirms this is not theoretical: rank 1 on efficiency is a
**0.938** submission, rank 3 is a **0.942** (5-fold DINOv2 @224px), and the overall LB
leader (0.952) sits at efficiency rank 55 — accuracy bought with runtime.

### State of the leaderboard (2026-08-21)

- #1 **0.952**, #10 **0.943**, #49 **0.934**; 2,133 teams.
- Forum estimates for final private: gold ≳ 0.94–0.975, winner ~0.96
  ([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735767)).
- No DICOM-metadata shortcut exists: header-only models reach 0.65 on random folds but
  **0.598 on scanner-grouped folds**; the 0.053 gap is *site memorisation*
  ([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517)).
  Consequence: **group folds by site/scanner fingerprint** or CV lies by ~0.05.

---

## 2. What the top single models are (forum evidence)

From ["Best single-model score"](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304):

| Who | Single-model public LB | Setup |
|---|---|---|
| Tucker Arrants (#8) | **0.934 → 0.942** | 5-fold, **224 px** |
| Scott Willis (#25) | 0.938 | single model, single fold |
| k256.dev (#77) | 0.926 | 5-fold @336 px, **LLM labels only** |
| Tim Krige (#90) | 0.92 (OOF = LB) | single model; "look at the images your model sees" |
| Tom Aindow (#49) | 0.915 | DINOv2; **392 px, 150 mm centre crop (0.383 mm/px), bag of 32 slices/study** |

And from ["Scaling the encoder bought us nothing"](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)
(paired, 3-fold, 2,652-study OOF, noise floor measured at 0.0020):

- **DINOv2-S → DINOv2-B: +0.0011.** 4× the parameters for nothing measurable. Only 5/12
  labels moved in Base's favour.
- Their real gains came from **crop geometry** (+0.0059, moved 10/12 labels) and
  **slice position**, not from capacity.
- **3 slices beat 9** by ~4× their noise floor (confounded with anchor position — treat as
  "fewer, better-placed slices", not a tuned number).
- Filename order matches anatomical order in only ~5% of series here; sort by
  `ImagePositionPatient · normal` (this repo already does — `kaggle_data.order_series`).
- FOV takes 71 distinct values (median 160 mm), so **crop to a fixed physical extent in mm
  before resizing** or the encoder sees knees at scales differing by a factor of several.
- The whole visual input of the model for 4,407 studies is **11.1 GiB uint8**
  (4,407 × 6 slots × 9 slices × 224²). Decoding costs 55 min/run; cached once it is free,
  and a fold then trains in **67 min on an M4 Pro (no GPU)** vs 76 min on a Kaggle T4.
- Swapping encoders without swapping their normalisation constants silently sinks the run
  (RAD-DINO trap). Read `preprocessor_config.json`.

### The labels are the ceiling

From ["'Not addressed' is a label too"](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)
and the public [RSNA Knee LLM Report Labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) (CC0):

| Label source | macro AUC vs the 58 gold studies |
|---|---|
| regex / lexicon extraction (what this repo's `src/labels.py` does, ~0.87 in-sample / 0.815 LOFO) | 0.814 |
| LLM v2 | 0.887 |
| **LLM v4 blend (two independent LLM readers)** | **0.893** |

- 25.4 % of cells are "the report does not address this" (= 0.5). Weight the loss by
  `2·|p − 0.5|` so those contribute nothing. Silence means *absent* for Baker's (3 % positive
  when silent) but *unknown* for Synovitis (34 %); v2 fills silent Synovitis from Effusion.
- The host has **explicitly permitted** hosted LLMs for label extraction
  ([ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)).
- The 58 gold studies are trauma-enriched (ACL 41 % vs ~20 % corpus) and contain verified
  mislabels; never set priors or tune on them (this repo already documented this).

---

## 3. What the literature says (2024–2026)

**2D backbone per slice + light aggregation beats 3D CNNs — consistently.**

- **Medical Slice Transformer** (Müller-Franzes et al., 2025,
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)): DINOv2-S per slice → one
  transformer layer with a CLS token over slices. Knee MRI (MRNet, 1,199 pts) meniscus tear
  **AUC 0.85 vs 0.69 for 3D ResNet**. 23M params total (22M DINOv2 + 1M head). **Fine-tuning
  the encoder was essential** — frozen DINOv2 collapsed to 0.62–0.66 on MRI/CT.
- **AnyMC3D** ([arXiv 2512.12887](https://arxiv.org/abs/2512.12887)): frozen 2D foundation
  model + LoRA (r=8) + **learnable-query attention pooling over slices**, per-view adapters
  for multi-plane/multi-sequence input, ~1.2–2.0 M trainable params per task. Avg AUROC
  over 10 3D tasks **0.894 vs 3D DenseNet 0.833, 3D ResNet 0.809, MST 0.869**; 3× the data
  efficiency of a 3D CNN; 1st place in VLM3D. Their appendix notes *every* recent 3D
  classification challenge was won by 2D/2.5D methods.
- **Co-plane attention (CoPAS)**, Nature Communications 2024
  ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)): 12 knee abnormalities,
  1,748 subjects, three plane branches with cross-plane and cross-sequence attention on
  224×224×24 volumes; avg AUC **0.812** (ACL 0.949, MCL 0.782, meniscus 0.763, effusion
  0.787, contusion 0.821). Useful as a per-label difficulty prior, and as evidence that a
  heavier 3D design does *not* reach the 0.93 the Kaggle 2.5D models already get.
- **23-condition knee MRI system**, European Radiology 2025
  ([link](https://link.springer.com/article/10.1007/s00330-025-12052-8)): 3,121 studies,
  ResNet18-style slice encoder → slice tokens + positional embedding → encoder-only
  transformer with CLS; external test on 448 studies; improved resident performance.
  Same shape as MST.
- **Segmentation-guided 3D** (Astuto et al., Radiology:AI 2021,
  [link](https://pubs.rsna.org/doi/full/10.1148/ryai.2021200165)): V-Net ROIs + 3D CNN on
  1,435 studies; AUC cartilage 0.93, meniscus 0.93, BME 0.83, ACL 0.90. Strong but needs
  segmentation labels we do not have — not a 62-day path.
- **KneeXNet** (Frontiers 2025): GCN + multi-scale 3D conv, MRNet AUC 0.985/0.972/0.968 —
  but 4×A100 for 48 h and a ≥16 GB GPU at inference. The opposite of efficient.

**Prior RSNA winners used exactly this family.**
[RSNA 2023 1st](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution): EffNetV2-S /
CoaT-lite on 2.5D triplets + GRU over slices, auxiliary segmentation loss +0.01–0.03.
RSNA 2024 1st ([analysis in AnyMC3D](https://arxiv.org/abs/2512.12887)): localize-then-
classify, ConvNeXt-S / EffNetV2-S on multi-view 2.5D stacks + BiLSTM + attention-MIL.
[RSNA 2024 2nd](https://github.com/brendanartley/RSNA-2024-Competition): encoder + LSTM +
attention pooling, `0.5·loss(labels) + 0.5·loss(pseudo-labels)`, one A100 for 24 h.

**LLM-extracted "silver" labels are now standard practice.** MICCAI 2025
([paper](https://papers.miccai.org/miccai-2025/0580-Paper3788.html)) shows modern LLMs
extract diagnostic labels from reports at >0.96 AUC and that training a 3D ResNet-18 on them
with label smoothing works; Decipher-MR ([arXiv 2509.21249](https://arxiv.org/html/2509.21249v1))
reports report-supervision gains on MRI tasks.

**MRI foundation models exist but are not the efficient choice yet.** MRI-CORE
([arXiv 2506.12186](https://arxiv.org/html/2506.12186v1), 6M slices / 110k volumes),
PRISM/MARS (Nature BME 2026, [arXiv 2508.07165](https://arxiv.org/abs/2508.07165), 336k
volumes, best on 39/44 benchmarks), OAI-DINO ([ISMRM 2025](https://archive.ismrm.org/2025/4014.html),
self-supervised on OAI knees, beats scratch and supervised pretraining), MedSigLIP
(400M, 448 px — too heavy). These are the **one worthwhile encoder ablation** after the
pipeline is right; the forum's RAD-DINO result (lost to DINOv2 until normalisation was fixed)
is the cautionary tale. **Distillation** (MedAlmighty, Frontiers 2025: DINOv2 → ResNet
student) is the fallback if we ever need to go below ViT-S for runtime.

**Industry.** The closest product to this task is Incepto's **KEROS** (CE Class IIa):
ACL, MCL, meniscal tears, cartilage, bone oedema, effusion, popliteal cysts — essentially
our label set ([product page](https://incepto-medical.com/product/keros)). A 2025
European Radiology reader study evaluated a knee AI trained on **23,074** studies
([link](https://link.springer.com/article/10.1007/s00330-025-11820-w)). Gleamer bought
Caerus (lumbar MRI) and Pixyl in 2025 to enter MRI; ImageBiopsy Lab's KOALA (FDA) grades
knee OA on X-ray; See All AI raised $33M (May 2025) for orthopaedic imaging. Takeaway:
commercial systems reach clinical grade with **tens of thousands** of report-labelled
studies — our 4,407 with LLM labels is the same recipe at small scale.

---

## 4. The recommended model

**Name it something and keep it small.** Working name: *SlotKnee-S*.

```
study ─► header pass (stop_before_pixels, ~0.2 ms/file): plane, TR/TE, fat-sat, IPP·normal, laterality
      ─► pick ≤6 sequence slots: SAG-FS, COR-FS, AX-FS, SAG-noFS, COR-T1, SAG-T1  (presence mask)
      ─► per slot: sort slices by geometry → 3 anchors (linspace, ends clipped) → 3 adjacent
         slices each = 3 triplets ──► crop fixed 140 mm around image centre → 224 px
         (0.625 mm/px; 252 px = 0.556 mm/px is the next honest step, still cheap)
      ─► laterality normalise (flip cor/ax, reverse sag stack; side from IPP x-sign when tag absent)
      ─► DINOv2-S/14 (timm vit_small_patch14_dinov2), last 4 blocks + norm trainable, LLRD,
         CLS+mean-patch pooled ──► 384-d embedding per triplet
      ─► mean over the slot's triplets ──► 6 slot tokens + learned slot identity
      ─► 12 learned queries, one per finding, masked attention over present slots
      ─► 12 logits
```

Why each piece:

| Choice | Evidence |
|---|---|
| DINOv2-S, fine-tuned partially | MST: fine-tuning essential; forum: Base = Small; Kaggle Models hosts DINOv2 offline |
| 224–252 px on a fixed-mm crop | forum winners at 224; Nyquist argument for 1 mm tears says ≤0.5 mm/px → 252–280 px is the only resolution step worth testing |
| 3 adjacent slices as RGB | RSNA 2023/2024 winners; "3 beat 9" on this corpus |
| per-finding attention over sequence slots | AnyMC3D's learnable-query pooling; pilkwang baseline head; each finding is read on particular sequences |
| LLM v4 labels, weight `2·p−1`, gold ×8 | 0.893 vs gold; "not addressed" handled; gold are the only image-read labels |
| site-grouped, report-hash-grouped folds | 0.053 leakage otherwise; identical template reports across studies |
| BCE (soft targets), no focal/ASL | this repo's own analysis: 5:1 prevalence, rank metric |
| 5 folds → **weight-soup or 2-fold average** | `src/soup.py` exists; inference cost is linear in members |

### Cost budget (per study, Kaggle T4, fp16)

| Stage | Cost | Basis |
|---|---|---|
| header pass | ~180 files × 0.2 ms ≈ 40 ms | measured here 0.16 ms/header |
| pixel decode, only the ~18–30 selected slices | 10–200 ms | 0.3–0.6 ms/slice uncompressed (measured); full data also has JPEG-lossless/J2K, budget ~5–10 ms/slice |
| DINOv2-S forward, 18–30 triplets | ~30 ms | ViT-S/14@224 ≈ 4.6 GFLOPs; ~1 ms/img fp16 on T4 |
| **Total** | **≈ 0.1–0.3 s/study → 2–7 min for 1,300 studies** | plus ~2 min import/model load |

That is a **~10-minute submission**, i.e. the efficiency runtime term ≈ 0.02 — equivalent
to 0.008 AUC. Every heavier design must buy back more than that in AUC to be worth it.

### Why *not* the alternatives

- **ConvNeXt-B / EffNetV2-M @384, 8 models** (this repo's current deploy): 88M + 53M
  params, 6–8 forwards per study at 384 px — an order of magnitude more compute for no
  measured gain at this n, and it trains on 649 local studies with regex labels.
- **3D CNNs / CoPAS-style**: 0.81 in the literature vs 0.93 for 2.5D on this exact task.
- **ViT-B/L, DINOv3, MedSigLIP**: null or unavailable (DINOv3 licence rejections reported);
  MedSigLIP is 400M at 448 px.
- **Big TTA ensembles**: public notebooks with 20+ models sit at ~0.91; the best singles at
  0.94.

---

## 5. What has to change in this repo (gap analysis)

Already right here (keep): geometric slice ordering; plane-balanced selection; laterality
guard on hflip; NaN-masked BCE with soft targets and per-cell weights (`labels_csv`,
`labels_weight`, `labels_weight_column`); valid-submission-first inference with measured
batch sizing and single-forward TTA; weight soup; offline timm weight loading.

Missing or wrong for this design:

1. **Training data**: only 649 of 4,407 studies are on disk locally (80 GB). Build the
   **11 GiB uint8 slot cache once on Kaggle** (fits the 20 GB notebook-output limit), save
   it as a dataset, then train every fold **locally on the M5 Pro** (the forum measured
   67 min/fold on an M4 Pro, no GPU quota) or on Kaggle GPU. This is the single biggest
   unlock: experiments cost an hour, not a day.
2. **Labels**: ingest `llm_labels_v4_blend.csv` through the existing `labels_csv` path with
   `labels_weight_column = 2·|p−0.5|`; keep `src/labels.py` as the fallback reader.
3. **Physical crop**: add a fixed-mm centre crop from `PixelSpacing` before resize
   (`kaggle_data.py` has none today).
4. **Slot selection by contrast**: use `Fluid_Sensitive`/`Fat_Suppression` (or TR/TE) to fill
   the six slots; today selection is plane-only.
5. **Head**: per-finding masked attention over slots (replace the pooled-channel head);
   ~0.2 M params.
6. **Folds**: GroupKFold on scanner fingerprint (Manufacturer + Model + Software +
   ImagingFrequency + Coil) and report hash; validate on *derived* labels across 2,600+
   studies, report gold agreement only as a sanity check.
7. **Inference**: header-first selection so only ~30 of ~180 files per study are pixel-
   decoded; fp16 `channels_last`; one forward per study for all triplets; drop the zoom TTA.
8. **Selection metric**: with 800+ derived-label validation studies per fold, `val_auc`
   becomes usable again (the repo's `auto` rule flips it on by itself).

---

## 6. Nine-week plan (today → Oct 22)

| Week | Deliverable | Gate |
|---|---|---|
| 1 (Aug 22–28) | Kaggle cache-builder notebook: header index + 6-slot × 9-slice uint8 cache for all 4,407 + test path; LLM labels wired; site-grouped folds | cache dataset ≤ 12 GB, decode ≤ 60 min |
| 2 | First SlotKnee-S 5-fold on the cache (224 px, last-4 blocks); submit | OOF ≥ 0.88 on derived labels, LB ≥ 0.90 |
| 3–4 | Ablations, paired, 2 seeds each: crop 130/140/160 mm; 224 vs 252 px; 1 vs 3 anchors; slot-attention vs mean; unfreeze depth 2/4/6 | keep only changes > 2× measured noise floor |
| 5 | Label ablation: v4 blend vs v2 vs regex; gold weight 4/8/16; silence handling per label | — |
| 6 | Encoder ablation (one): OAI-DINO or MRI-CORE **with correct normalisation** | accept only if > noise floor |
| 7 | Soup vs 2-fold vs 5-fold at fixed runtime; fp16 + header-first decode; runtime measured in a Kaggle run | total ≤ 10 min |
| 8 (Oct 10–16) | Freeze; pseudo-label round on test-like unlabeled train rows if time (RSNA'24 2nd recipe) | — |
| 9 (Oct 17–22) | Final two selections: best-AUC and best-efficiency (may be the same) | — |

Measurement protocol throughout (from this repo's own hard-won rule): same code revision,
≥2 seeds, compare pooled OOF on derived labels with a stated noise floor; a single 5-fold
run cannot separate two configurations.

---

## Sources

Competition: [overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) ·
[data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) ·
[leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) ·
[efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) ·
[RSNA announcement](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri) ·
[RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)

Forum: [best single model](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) ·
[scaling the encoder bought nothing](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154) ·
["not addressed" is a label](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932) ·
[metadata shortcut probe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517) ·
[LLM use ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) ·
[final ceiling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735767) ·
[DINOv3 licence](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733313) ·
[58 labels / languages](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106) ·
[pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) ·
[LLM labels dataset](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

Papers: [Medical Slice Transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/) ·
[AnyMC3D](https://arxiv.org/abs/2512.12887) ·
[CoPAS, Nat Commun 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/) ·
[23-condition knee system, Eur Radiol 2025](https://link.springer.com/article/10.1007/s00330-025-12052-8) ·
[Astuto et al., Radiology:AI 2021](https://pubs.rsna.org/doi/full/10.1148/ryai.2021200165) ·
[KneeXNet](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full) ·
[RSNA 2023 1st](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution) ·
[RSNA 2024 2nd](https://github.com/brendanartley/RSNA-2024-Competition) ·
[LLM labels for pretraining, MICCAI 2025](https://papers.miccai.org/miccai-2025/0580-Paper3788.html) ·
[Decipher-MR](https://arxiv.org/html/2509.21249v1) ·
[MRI-CORE](https://arxiv.org/html/2506.12186v1) ·
[PRISM / multi-sequence pretraining](https://arxiv.org/abs/2508.07165) ·
[OAI-DINO, ISMRM 2025](https://archive.ismrm.org/2025/4014.html) ·
[MedSigLIP](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card) ·
[MedAlmighty distillation](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1527980/full) ·
[FM utility in MSK MRI, npj Digit Med 2026](https://arxiv.org/abs/2501.13376)

Industry: [Incepto KEROS](https://incepto-medical.com/product/keros) ·
[knee AI reader study, Eur Radiol 2025](https://link.springer.com/article/10.1007/s00330-025-11820-w) ·
[Gleamer acquires Pixyl & Caerus](https://www.gleamer.ai/press/gleamer-announces-acquisitions-of-pixyl-and-caerus-medical) ·
[ImageBiopsy Lab KOALA FDA](https://www.dotmed.com/news/story/49249/) ·
[See All AI $33M](https://bonezonepub.com/2026/02/04/six-orthopedic-startups-to-track-in-2026/) ·
[ScanDiags FDA (lumbar)](https://www.fda.gov/media/178541/download)
