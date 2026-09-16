# SlotKnee-S — efficiency benchmark & design review (module R)

*2026-08-22.  Everything in §1 is MEASURED on this machine (Apple M5 Pro, 24 GB, MPS,
torch 2.8.0, timm 1.0.28, pydicom 2.4.4, cv2 5.0.0, python 3.9) unless a column or row is
explicitly marked **est.** — the T4 projection method and its assumptions are stated in
§1.4.  The objective is `J = macroAUC − seconds/71,700` (0.01 AUC ≡ 717 s) for ~1,300 test
studies on a Kaggle T4; §2 ranks changes by expected ΔJ; §3 lists defects with file:line.*

Commands used (both < 3 min):

```
python3 scripts/bench_slotknee.py --quick                                        # §1.1–1.4
python3 scripts/bench_slotknee.py --quick --train-probe --no-alt --no-cpu-ref \
        --n-studies 1 --P 224 --token-proxy-P                                    # §1.5
```

The benchmark is standalone (own header/decode/encoder harness following spec §3–§4); §1.6
cross-checks it against the landed `src/slots.py` / `scripts/build_slot_cache.py`.

## 1. Benchmark

### 1.1 Header pass + decode, per study (P=224, G=3, T=3, crop 140 mm, 4 real studies)

| study | files | hdr ms (12 tags) | hdr ms (all) | hdr ms (8KB head) (miss) | slots | decoded | decode ms | ·read | ·resize | ·norm | decode ms 2-step | G=1 ms | G=2 ms | T=1 ms | T=5 ms | P=252 ms | FULL decode ms | crop fb | shape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 56622764 | 419 | 112.7 | 109.8 | 65.0 (0) | 6 | 54 | 74.5 | 14.4 | 45.8 | 14.3 | 53.8 | 24.7 | 50.2 | 25.4 | 137.1 | 83.7 | 200.3 | 0 | 640x640 |
| 04016339 | 198 | 54.2 | 53.1 | 31.3 (0) | 6 | 54 | 130.4 | 16.2 | 103.1 | 11.1 | 37.9 | 43.6 | 87.8 | 44.0 | 231.1 | 161.2 | 103.8 | 9 | 512x512 |
| 56266866 | 136 | 40.7 | 40.7 | 22.9 (0) | 4 | 36 | 74.8 | 16.2 | 48.9 | 9.6 | 63.1 | 21.3 | 42.0 | 20.8 | 115.5 | 78.1 | 68.0 | 0 | 384x384 |
| 66582736 | 143 | 31.4 | 30.0 | 19.6 (0) | 4 | 36 | 37.8 | 8.8 | 20.5 | 8.5 | 33.3 | 13.5 | 25.7 | 12.9 | 73.6 | 44.0 | 55.3 | 0 | 704x640 |

Mean: 224 files/study; **header 59.8 ms** (0.267 ms/file; all-tags is the same — pydicom's
`specific_tags` does not speed the parse; an 8 KB partial read is **34.7 ms** with 0/896
files missing a geometry tag); **decode of 45 slices 79.4 ms** (1.76 ms/slice = read 13.9 +
resize 54.6 + percentile-norm 10.9); **two-step resize 47.0 ms** (resize part 54.6→23.3);
G=1 25.8 / G=2 51.4 / T=1 25.8 / T=5 139.3 / P=252 91.7 ms; FULL decode of every file
107 ms (0.48 ms/file — local data is uncompressed Explicit VR LE); Laterality tag in 2/4
studies (geometry fallback needed); "decoded" is 6·G·T over *present* slots (54 when all six
exist — not the spec's "27", see defect S1).

Two decode facts that drive §2.4:

* `cv2.INTER_AREA` on float32 is pathological at **non-integer** ratios (measured on this
  machine, single 2-D resize): 478→224 = **1.04 ms**, 640→224 = 0.53 ms, 336→224 = 0.32 ms,
  but 448→224 (exact 2×) = **0.015 ms**.  A two-step resize (integer-factor INTER_AREA →
  INTER_LINEAR remainder, ratio < 2) is equivalent output at 224 and cuts the resize cost
  ~2.3× end-to-end.
* pydicom header parse: 0.15–0.27 ms/file warm; reading only the first 8 KB into `BytesIO`
  then `dcmread(force=True)` is ~40 % cheaper and lost no geometry tags on 896 local files
  (a fallback to the full read is still mandatory — vendor headers can exceed 8 KB).

### 1.2 DINOv2-S encoder forward (`vit_small_patch14_dinov2.lvd142m`, images/study = 6·G)

| dev | P | dtype | B | G | chunk | imgs | ms/batch | ms/study | ms/img | finite | max\|Δ\| vs fp32 | mps driver MiB | s/1300 (this dev) | T4 ms/study est. | T4 s/1300 est. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mps | 224 | fp32 | 1 | 3 | 64 | 18 | 46.6 | 46.6 | 2.59 | yes | n/a | 1215 | 60.6 | 179.96 | 233.9 |
| mps | 224 | fp16-autocast | 1 | 3 | 64 | 18 | 21.6 | 21.6 | 1.20 | yes | 0.033 | 1223 | 28.1 | 20.86 | 27.1 |
| mps | 224 | fp16-half | 1 | 3 | 64 | 18 | 19.2 | 19.2 | 1.07 | yes | 0.030 | 1239 | 25.0 | 18.59 | 24.2 |
| mps | 252 | fp32 | 1 | 3 | 64 | 18 | 63.9 | 63.9 | 3.55 | yes | n/a | 1291 | 83.0 | 246.65 | 320.6 |
| mps | 252 | fp16-autocast | 1 | 3 | 64 | 18 | 28.9 | 28.9 | 1.60 | yes | 0.034 | 1307 | 37.5 | 27.86 | 36.2 |
| mps | 252 | fp16-half | 1 | 3 | 64 | 18 | 25.5 | 25.5 | 1.42 | yes | 0.033 | 1315 | 33.2 | 24.64 | 32.0 |
| mps | 224 | fp32 | 4 | 3 | 64 | 72 | 188.9 | 47.2 | 2.62 | yes | n/a | 2437 | 61.4 | 182.43 | 237.2 |
| mps | 224 | fp16-half | 4 | 3 | 64 | 72 | 74.6 | **18.6** | **1.04** | yes | 0.042 | 1413 | 24.2 | 18.00 | **23.4** |
| mps | 224 | fp16-half | 4 | 3 | 18 | 72 | 76.9 | 19.2 | 1.07 | yes | 0.057 | 1413 | 25.0 | 18.56 | 24.1 |
| mps | 252 | fp32 | 4 | 3 | 64 | 72 | 253.9 | 63.5 | 3.53 | yes | n/a | 2593 | 82.5 | 245.14 | 318.7 |
| mps | 252 | fp16-half | 4 | 3 | 64 | 72 | 97.7 | 24.4 | 1.36 | yes | 0.037 | 1569 | 31.7 | 23.58 | 30.6 |
| mps | 252 | fp16-half | 4 | 3 | 18 | 72 | 99.1 | 24.8 | 1.38 | yes | 0.035 | 1569 | 32.2 | 23.92 | 31.1 |
| mps | 224 | fp32 | 4 | 3 | 18 | 72 | 192.0 | 48.0 | 2.67 | yes | n/a | 1569 | 62.4 | 185.39 | 241.0 |
| mps | 224 | fp16-autocast | 4 | 3 | 64 | 72 | 87.9 | 22.0 | 1.22 | yes | 0.036 | 1585 | 28.6 | 21.21 | 27.6 |
| mps | 224 | fp16-half | 1 | 1 | 64 | 6 | 6.7 | 6.7 | 1.11 | yes | n/a | 1583 | 8.7 | 6.43 | 8.4 |
| mps | 224 | fp32 | 1 | 1 | 64 | 6 | 16.3 | 16.3 | 2.71 | yes | n/a | 1583 | 21.1 | 62.80 | 81.6 |
| mps | 224 | fp16-half | 1 | 2 | 64 | 12 | 12.7 | 12.7 | 1.06 | yes | n/a | 1531 | 16.5 | 12.24 | 15.9 |
| mps | 224 | fp32 | 1 | 2 | 64 | 12 | 31.4 | 31.4 | 2.61 | yes | n/a | 1531 | 40.8 | 121.14 | 157.5 |
| mps | 224 | fp16-half | 4 | 1 | 64 | 24 | 25.2 | 6.3 | 1.05 | yes | n/a | 1531 | 8.2 | 6.08 | 7.9 |
| mps | 224 | fp32 | 4 | 1 | 64 | 24 | 62.7 | 15.7 | 2.61 | yes | n/a | 1531 | 20.4 | 60.55 | 78.7 |
| mps | 224 | fp16-half | 4 | 2 | 64 | 48 | 50.3 | 12.6 | 1.05 | yes | n/a | 1532 | 16.4 | 12.14 | 15.8 |
| mps | 224 | fp32 | 4 | 2 | 64 | 48 | 125.0 | 31.2 | 2.60 | yes | n/a | 1532 | 40.6 | 120.65 | 156.8 |
| mps | 196* | fp16-half | 4 | 3 | 64 | 72 | 56.9 | 14.2 | 0.79 | yes | n/a | 1586 | 18.5 | 13.73 | 17.9 |
| mps | 168* | fp16-half | 4 | 3 | 64 | 72 | 41.1 | 10.3 | 0.57 | yes | n/a | 2682 | 13.4 | 9.92 | 12.9 |
| cpu | 224 | fp32 | 1 | 3 | 64 | 18 | 225.1 | 225.1 | 12.51 | yes | n/a | 214 | 292.6 | n/a | n/a |

\* P=196/168 rows are **token-count proxies** for token pruning / a lower-resolution slot
(196 patches = −24 %, 144 = −44 % vs 256), a cost measurement only, not a quality claim.

Readings: fp16 is **2.5× fp32 on MPS** (both `.half()` and autocast; `.half()` slightly
faster); `encoder_chunk` 18 vs 64 is noise (≤ 3 %); batching 4 studies buys ~3 %; P=252
costs +31 % model time; cost is linear in G at fixed ms/img.  fp16 CLS-feature deviation vs
fp32 ≤ 0.057 on features of O(1) magnitude, all finite.  Peak RSS of the whole run 1.40 GiB
(< 4 GB).  A measurement lesson baked into the script: a forward WITHOUT
`inference_mode/no_grad` retains ~128 MiB/img of autograd state on MPS (72 imgs → 15.7 GiB
driver, 40× slower) — any inference loop must run under `torch.inference_mode()`.

### 1.3 Alternative encoders (record only; random init; same 72-image workload, P=224)

| encoder | params M | fp16 ms/img | fp16 ms/study | fp32 ms/img | T4 s/1300 est. (fp16) | J cost (AUC-equiv) |
|---|---|---|---|---|---|---|
| vit_small_patch14_dinov2 (ref) | 21.6 | 1.04 | 18.6 | 2.62 | 23.4 | 0.0003 |
| vit_tiny_patch16_224 | 5.5 | 0.32 | 5.8 | 0.71 | 7.3 | 0.0001 |
| efficientnet_b0 | 4.0 | 0.59 | 10.7 | 0.96 | 13.4 | 0.0002 |
| convnext_nano | 15.0 | 0.75 | 13.4 | 1.48 | 16.9 | 0.0002 |
| mobilenetv3_small_100 | 1.5 | 0.16 | 2.9 | 0.22 | 3.6 | 0.0001 |

DINOv2-S with 2 vs 4 trainable blocks has **identical inference cost** (same forward; only
training memory/time differ — §1.5).  The CNN T4 numbers reuse the ViT calibration and are
the weakest estimates here; use the MPS ratios.  The whole ViT-S model costs 0.0003
AUC-equivalent — a smaller student can recover at most that (see §2.12).

### 1.4 Per-study budget and 1,300-study projection

| scenario | hdr ms | decode ms | model ms | total ms/study | s/1300 serial | s/1300 overlapped | J cost serial |
|---|---|---|---|---|---|---|---|
| this machine (MPS, measured) | 59.8 | 79.4 | 18.6 | 157.8 | 205.1 | 180.9 | 0.0029 |
| this machine, 8KB header + 2-step resize (measured) | 34.7 | 47.0 | 18.6 | 100.3 | 130.4 | 106.2 | 0.0018 |
| T4 est., uncompressed DICOM | 149.4 | 198.5 | 18.0 | 365.9 | 595.6 | 572.2 | 0.0083 |
| T4 est., 8KB header + 2-step resize | 86.7 | 117.5 | 18.0 | 222.3 | 408.9 | 385.5 | 0.0057 |
| T4 est., all 45 slices JPEG-lossless | 149.4 | 513.5 | 18.0 | 680.9 | 1005.1 | 981.7 | 0.0140 |
| T4 est., FULL decode (no header-first) | 149.4 | 267.1 | 18.0 | 434.5 | 684.8 | 661.4 | 0.0096 |

**T4 projection method (all "est." values).**  Model: calibrate k = 1.0 ms/img (the brief's
T4 fp16 ViT-S/14@224 figure, consistent with 4.6 GFLOPs at ~4.6 TFLOPS achieved) ÷ measured
MPS fp16-half @224 B=4 (1.04 ms/img) → k = 0.966; every MPS model number is scaled by k;
fp32 rows additionally ×4.0 (T4: 8.1 TFLOPS fp32 vs 65 fp16 tensor-core; ViT-S realises
~4×).  CPU stages (pydicom + cv2): ×2.5 for one Kaggle vCPU vs one M5 Pro P-core (assumed,
not measured — Kaggle Xeons at 2.0–2.3 GHz vs Apple P-core; the true factor is probably
2–3×).  +120 s fixed (container, imports, checkpoint load).  JPEG-lossless decode budgeted
at 7 ms/slice (brief §4: 5–10; not measurable locally — all local files are uncompressed).
"Overlapped" = max(CPU, GPU) per study — one decode worker feeding the GPU.
**Headline: the GPU is 12 % of the serial budget; the CPU decode path and the fixed
overhead are everything else.**  If the hidden test set is JPEG-compressed, header-first
selection is worth ~5× more (last two rows: decoding every file at 7 ms would add ~2,000 s).

### 1.5 Training probe (fwd+bwd, fp32 on MPS, 6·G·bs images, AdamW step; probe = encoder
+ linear head, no slot attention — the real model adds < 1 ms)

| bs | trainable blocks | imgs/step | trainable params M | ms/step | mps driver MiB | RSS MiB | s/epoch (4,407) est. |
|---|---|---|---|---|---|---|---|
| 4 | 2 | 72 | 3.6 | 271 | 2,326 | 720 | 298 |
| 4 | 4 | 72 | 7.1 | 351 | 4,478 | 889 | 387 |
| 8 | 4 | 144 | 7.1 | 755 | 7,636 | 889 | 416 |

bs 8 / 4 blocks / no checkpointing: **7.6 GiB MPS + 0.9 GiB RSS, ≈ 7 min/epoch → ≈ 42
min/fold (6 epochs)** on the M5 Pro — consistent with the forum's 67 min/fold on an M4 Pro.
A separate measurement taken while `src/slotknee.py` was being built put bs 8 *with*
grad-checkpointing at 1.25 s/step and 4.2 GiB: checkpointing
trades +65 % step time for −45 % memory — unnecessary at 24 GB, keep it for a 16 GB T4 at
bs ≥ 16.  Extrapolation (activation-dominated, linear in bs·blocks): bs 16/4 blocks ≈ 15 GiB
— do not go there on the T4 without checkpointing.

### 1.6 Cross-check against the landed modules (measured)

* `src.slots.index_study` on the same 4 studies: 50.3 ms/study (vs 59.8 for my all-files
  harness — it survey-reads one header per series, then geometry-reads only the chosen
  series; here 0.92× of all files because a 160-slice series was chosen, see defect A1);
  `build_study_tensor` decode 92.5 ms; total build 155 ms/study.
* `scripts/build_slot_cache.py` real run (its own log, `cache/slots_P224/train_index.json`):
  **649 studies in 27.7 s with 4 workers** (23.5 studies/s; 148 ms/study per worker), 28,377
  slices decoded, 0 failures, 909 crop fallbacks, worker peak RSS 0.37 GB, 1.76 GB on disk.
  At that rate the full 4,407-study cache is ~3.2 min locally; on Kaggle (2 vCPU, compressed
  transfer syntaxes) budget 45–90 min — within the week-1 gate.
* Slot presence over all 4,407 studies (train_series.csv): SAG_FS 94.2 %, COR_FS 96.4 %,
  AX_FS 100 %, SAG_T1 96.8 %, COR_T1 77.3 %, **AX_T1 19.4 %**; mean 4.84/6 slots → 19.3 %
  of a dense cache is zero padding.  Cache sizes at 4,407 studies: P224/G3/T3 dense
  **11.94 GB**, ragged (present slots only) 9.63 GB, G=2 7.96 GB, P=196 9.14 GB, P=252
  15.11 GB.

## 2. Design review — ranked by expected ΔJ

Noise floor for AUC claims: 0.0020 (forum, 2,652-study OOF).  Runtime ΔJ figures use §1.4.

1. **Train on LLM labels over all 4,407 studies (not regex over 649).  ΔJ ≈ +0.02…+0.06 —
   an order of magnitude above everything else.**  Evidence: label-vs-gold macro AUC 0.814
   (regex) vs 0.887 (LLM v2) vs 0.893 (v4 blend); a 0.926 public-LB single model trained on
   LLM labels only; 6.8× more studies.  Memory: none.  Risk: the CSVs are Kaggle datasets
   (attach offline); plus the v4 weighting subtlety — **v4_blend does not use the 0.5
   silence convention** (silence lands at 0.25/0.75 because it averages v2 with a hard 0/1
   reader), so `cell_weights` gives unaddressed cells weight 0.5 pulled toward 0.25 instead
   of 0.  That is a *different* (weakly-negative-for-silence) supervision than the brief's
   `2|p−0.5|` design.  Action: ablate (a) v4 as-is, (b) v2 + `fill_silent_synovitis`
   (exact 0.5 convention), (c) `load_llm_labels([v2, hard_reader], with_conf=True)`;
   `min_weight≈0.6` can gate v4's silence cells to 0 without touching decisive cells.
2. **Keep scanner+report grouped folds and select on derived-label OOF (spec §5, landed
   `src/folds.py`).**  ΔJ: indirect but decisive — random folds overstate CV by ~0.05 (site
   memorisation, forum probe), so every other ablation in this list would be judged on
   noise.  Cost: one header/study (~0.4 ms).  Risk: none.  This is the insurance policy for
   the whole §2.
3. **fp16 at T4 inference (and fix defect D2 so it actually engages).  ΔJ ≈ +0.0030.**
   Model 237 s → 23 s per 1,300 (est. from measured 2.5× MPS speedup and the ×4 T4 factor).
   Measured fp16 feature deviation ≤ 0.057, all finite.  Memory: −45 % activations.  Risk:
   fp16 overflow in exotic inputs — verify max|logit Δ| vs fp32 on ~20 studies once, then
   ship fp16.
4. **Cheapen the CPU decode path — it is 5× the GPU cost.  ΔJ ≈ +0.0026 (serial), +0.0029
   with one prefetch thread.**  Three measured pieces: two-step resize (79.4 → 47.0
   ms/study; the INTER_AREA non-integer-ratio pathology in §1.1), 8 KB partial header read
   with full-read fallback (59.8 → 34.7 ms), and overlapping decode with the GPU (serial →
   max()).  Memory: none.  Risks: the two-step resize changes pixel values slightly vs
   one-shot INTER_AREA — **train and infer must use the same resizer** (rebuild the cache
   with it); the 8 KB read needs the miss-fallback (0 misses on 896 local files, unknown
   vendors on Kaggle).  Implementation lives in `src/slots.py:_decode_crop` /
   `_survey_series` + `kaggle_data.order_series`'s per-file reads.
5. **Size the ensemble with J, not habit: each extra ViT-S member costs only 0.0003
   AUC-equivalent (23 s).**  Evidence: §1.2; the decode (the expensive part) is shared
   across members; efficiency-LB rank 3 is a *5-fold* DINOv2@224.  Plan: soup the 5 folds
   (`src/soup.py`) into one model (free), and rank-average it with 1–2 raw folds only if
   OOF shows each member adds > 0.0003 AUC (they usually add ~0.001–0.003).  Memory: 3
   resident ViT-S fp16 ≈ 130 MB.  Risk: soup needs a common init/trajectory — satisfied
   (same DINOv2 init, same schedule).  Do NOT drop to a single fold to save 23 s.
6. **Keep G=3 anchors in the cache; do not buy runtime with G=1.  ΔJ of G=1 ≈ +0.0025
   runtime but a likely larger AUC loss.**  Measured: G=1 saves 53.6 ms decode + 12.3 ms
   model per study.  But 4/12 labels are compartment-specific (Medial/Lateral Meniscus,
   Medial/Lateral OA) and a single mid-stack sagittal/coronal anchor never shows the
   medial and lateral compartments (the forum's "crop geometry and slice position" gains,
   +0.0059, argue placement matters more than count).  Ablate G=2 (saves ~half) properly
   (2 seeds, grouped OOF) before ever shipping it.
7. **Keep P=224.  P=252 costs ΔJ ≈ −0.0007 runtime and +3.2 GB cache; adopt only on a
   > 2×-noise-floor OOF gain.**  Measured: model +31 % (18.6→24.4 ms), decode 79→92 ms,
   cache 11.94→15.11 GB.  The Nyquist argument (0.556 mm/px) makes it the one resolution
   worth *testing* — as a paired ablation in week 3–4, not a default.
8. **Keep T=3.  T=1 saves ≈ +0.0019 runtime ΔJ and 8 GB of cache but discards the 2.5D
   context every recent RSNA winner used; T=5 costs −0.0021 ΔJ and a 19.9 GB cache (over
   the 20 GB output cap with anything else in the dataset).**  Measured decode: 25.8 / 79.4
   / 139.3 ms for T=1/3/5; encoder cost is T-independent (channels merge in patch embed).
9. **Cache strategy: 11.94 GB dense fits the 20 GB cap — ship dense now; halve later only
   if needed.**  Measured options that lose zero information: ragged layout skipping absent
   slots (−19.3 % → 9.63 GB; AX_T1 alone is 518/649 missing locally, 80.6 % globally) and
   per-shard zstd/npz lossless (~×1.8–2 → ~5–6 GB; +1–2 ms/study decompress at read).  G=2
   (7.96 GB) and P=196 (9.14 GB) also halve-ish but touch AUC (items 6–7).  Keep
   shard_size=512 (resumable, partial-upload friendly).  Building on Kaggle: measured 23.5
   studies/s × 4 workers locally → budget 45–90 min for 4,407 on 2 vCPU with compressed
   syntaxes.
10. **encoder_chunk: keep 64.**  Measured 18 vs 64: ≤ 3 % (noise); chunking exists to bound
    activation memory (1.4–1.6 GiB driver at chunk 64, fp16) — both fit everywhere; smaller
    chunks only add launch overhead.
11. **Token pruning / low-res axial slot: reject.**  Proxy measurements (P=196/168 = −24 %
    / −44 % tokens): best case saves ~10 s per 1,300 ≈ 0.0001 ΔJ, for pos-embed
    interpolation risk and real code complexity.  The same argument kills attention-map
    token dropping and per-slot resolutions.
12. **A smaller encoder can never pay under J here: reject (record kept in §1.3).**  The
    entire ViT-S forward is 23 s/1,300 = 0.0003 AUC-equivalent; even a *zero-cost* encoder
    recovers at most that, while the evidence says capacity below S loses more (S→B was
    +0.0011 — the curve is flat *above* S, not below; frozen-DINOv2 collapse in MST shows
    the features matter).  Distillation to ViT-Ti (0.32 ms/img) is a fallback only if the
    rules ever force CPU-only inference — there it would matter (CPU ViT-S is 12.5 ms/img).
13. **Trainable blocks 2 vs 4: an experiment-throughput knob, not a J knob.**  Inference
    is identical (§1.3 note).  Training (§1.5): 2 blocks is 23 % faster and half the
    memory; if a paired OOF shows ≤ noise-floor difference, run the week-3/4 ablation grid
    at 2 blocks and the final models at whichever won.  Grad-checkpointing off on the M5
    Pro (bs 8 fits in 7.6 GiB); on for T4 bs ≥ 16.
14. **Batch studies at inference (bs 4) and reuse the decode across checkpoints** — the
    landed two-pass infer script already achieves the reuse via a disk cache (3.5 GB for
    1,300 studies — fine).  Add the every-200-studies partial submission write (defect D4)
    so a late crash still submits.

## 3. Defects & spec inconsistencies (file:line)

**Spec (docs/slotknee_spec.md)**

* S1 — §3.6 "≈ 27 decoded 640² int16 slices ≈ 22 MB" and §7 "~27 slices":
  6·G·T = **54** at G=T=3 (measured; the brief's own 11.1 GiB cache math uses 54).  The
  echo at `scripts/build_slot_cache.py:17` ("~27 raw slices") is likewise wrong.  Budget
  ~44 MB/study of raw int16, or say "6·G·T (54; ~45 after missing slots)".
* S2 — §1 "no process may hold more than ~4 GB RSS": on macOS **RSS is blind to
  MPS/Metal memory** (measured: RSS 520 MiB while the process held 15.7 GiB of driver
  memory).  Any guard must also read `torch.mps.driver_allocated_memory()` /
  `torch.cuda.max_memory_reserved()`.  (`torch.mps` has no peak counter; driver-allocated
  after sync is the usable proxy.)
* S3 — §3.1's contrast fallback "fat-sat / STIR / T2 / PD → fluid-sensitive" contradicts
  the data: `Fluid_Sensitive == Fat_Suppression` on 24,371/24,371 series-CSV rows
  (verified).  The header rule that landed in `src/slots.py` (fat-sat tokens only, 98.0 %
  agreement) is right; the spec text should be amended to match `src/slots.py:82-84`.
* S4 — §3.6 describes a single flat `train_x.u8`; the landed cache shards
  (`train_x.001.u8`, `shard_size=512`, reader handles both).  Update the spec, keep the
  sharding (resumability + partial uploads).
* S5 — §4 says "All B·6·G images go through the encoder"; the landed model skips absent
  slots (`encode_absent=False`), a good deviation worth ~19 % of encoder images on average
  (mean 4.84/6 slots present).  Document it.

**src/slots.py (module A)**

* A1 — `src/slots.py:401-409` ("most slices wins", per spec): confirmed to pick **3-D
  acquisitions over the routine 2-D TSE** — study …622764's AX_T1 slot chose a 160-slice
  series (its other candidates: 16–32 slices).  Consequences: a different contrast family
  inside the T1 slots (3-D GRE/VIBE vs TSE), thin-slice triplets spanning ~1–3 mm instead
  of ~9–12 mm, and 160 extra geometry header reads (~27 ms).  Suggestion (already flagged
  when `src/slots.py` landed): among candidates prefer series with 12 ≤ n ≤ 60 (or SliceThickness ≥ 2 mm),
  falling back to most-slices; ablate before/after — expected small AUC effect, small
  runtime win, low risk.
* A2 — `src/slots.py:525` (`cv2.INTER_AREA` one-shot on float32): the single most
  expensive CPU operation in the pipeline at non-integer ratios (§1.1; 54.6 of 79.4
  decode-ms).  Two-step resize halves decode; see §2.4 (and keep cache/infer consistent).
* A3 — `src/slots.py:537` (`np.percentile` over the full G·T·P·P float32): 10.9 ms/study.
  Subsampling every 2nd–4th pixel for the 1–99 pct estimate cuts it ~4× with no visible
  effect on an 8-bit quantisation.  Minor.
* A4 — `src/slots.py:331` (`files[len(files)//2]` survey read): good choice (centre slice
  is the right laterality/geometry witness); noting it here because the spec's
  `.spacing per series` promise is fulfilled from ONE slice — mixed-resolution series
  (rare) would mis-crop; the crop_window fallback covers the failure mode.

**src/slotknee.py (module B)**

* B1 — `src/slotknee.py:61` vs `:185`: `_REQUIRED_ENCODER_ATTRS` omits `pos_drop`, which
  `_frozen_modules` dereferences — an exotic backbone fails with a bare AttributeError
  instead of the friendly TypeError.  Cosmetic.
* B2 — `src/slotknee.py:334`: `checkpoint_seq(..., use_reentrant=False)` — verified
  present in timm 1.0.28's signature; fine (and the non-reentrant choice is required after
  the `no_grad` prefix, as the comment says).
* B3 — module is otherwise clean and matches the spec; the `encode_absent=False` deviation
  is an efficiency win (S5).

**scripts/train_slotknee.py (module D)**

* D1 — **`scripts/train_slotknee.py:207-210`: validation crashes on soft targets.**
  `roc_auc_score(y_true=…)` receives the *soft* y (regex-calibrated values like 0.0066,
  0.93; LLM blends like 0.25/0.75) and raises `ValueError: continuous format is not
  supported` (reproduced directly on `regex_fallback` output).  Fix: binarise
  `y_true = (m_y[active, i] > 0.5)` (and skip cells where that leaves one class).  Until
  fixed, epoch-1 validation kills every training run that uses derived labels.
* D2 — `scripts/train_slotknee.py:125` + `scripts/infer_slotknee.py:31,68,125`: on MPS the
  autocast context is created with `device_type="cpu"`, so the intended fp16 inference
  **silently runs fp32 on MPS** (T4/cuda path is correct).  Local timing/efficiency
  numbers from this script under-report the fp16 speed by ~2.5×; use
  `device_type=device.type` (or `model.half()`).
* D3 — `scripts/train_slotknee.py:317` (`RunningWeightAverage`): the spec asked for EMA
  decay 0.998 per step; the landed class is a per-epoch uniform SWA (`src/train.py:1700`).
  Defensible (arguably better late in training) but it is a deviation — say so in the
  training log, and don't call it EMA.
* D4 — `scripts/infer_slotknee.py:167-171`: the final write **rebuilds submission.csv from
  processed studies only**.  Any study missing from `test_images/` or lost to an exception
  leaves a short (invalid) submission, defeating the copy-sample-first step; and the
  spec's "overwrite every 200 studies" partial write is absent.  Fix: merge predictions
  into the seeded sample frame by UID and rewrite it (and do so every N batches).
* D5 — `scripts/train_slotknee.py`: no per-fold resume, `--amp` parsed but ignored, and no
  peak-RSS/MPS-memory print per epoch (spec §6).  Minor.
* D6 — `scripts/smoke_slotknee.sh:11`: `rm -rf "$CACHE_DIR"` deletes the shared full
  `cache/slots_P224` (the 649-study, 1.76 GB artefact) every time the smoke test runs —
  use a dedicated `cache/slots_smoke` dir.  Also `:47` `len(df) == len(sample) or
  len(df) > 0` is a tautology (any non-empty frame passes); it hides defect D4.

**src/llm_labels.py, src/folds.py (module C)** — no functional defects found; both match or
exceed the spec (confidence-column support, `ImagingFrequency` kHz rounding with evidence,
empty-report handling).  The v4 silence convention is *documented* at
`src/llm_labels.py:39-45` — the ranked item §2.1 carries the action.

## 4. Bottom line

The model is already almost free: ViT-S fp16 @224 is **18.6 ms/study measured on MPS
(≈ 18 ms est. on T4) = 23 s per 1,300 = 0.0003 AUC-equivalent**.  The serial T4 budget is
~600 s est. (J ≈ 0.0083), of which ~75 % is pydicom+cv2 on the CPU and ~20 % fixed
overhead; the measured CPU-path fixes bring it to ~410 s serial / ~385 s overlapped
(J ≈ 0.0057).  Spend AUC-side effort on labels, folds, and anchor placement; spend
runtime-side effort only on the decode path; and size the ensemble knowing each extra
ViT-S member costs 0.0003.
