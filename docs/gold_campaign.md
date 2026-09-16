# Gold campaign schedule — SlotKnee-S, 2026-08-24 → 2026-10-22

Written 2026-08-23. Sources: the campaign log (cited below as "Log <time>", all 08-23 unless dated) and `docs/slotknee_runbook.md`.
Position at writing: g10t1 5-fold pooled OOF **0.8531** (Log 20:28), two submissions pending
(refs 55722433 / 55725679, Log 20:28 / 23:30), reference LB 0.835. SCORED 23:56 / 00:31: **0.866** (CPU, 2 g10t1 members, no TTA) and **0.864** (T4, 10-member multi-view + TTA) — pure coverage beats the equal-weight multi-view on the LB exactly as the OOF predicted (0.8531 vs 0.8457). Current best LB 0.866; folds-0/1 OOF 0.8513 → offset ≈ +0.013.
Target ~0.95 by Oct 22; phase definitions and standing policies are Log 23:40 (this doc schedules them).
Gate policy (Log 17:15): noise 0.0038 pooled; **Δ>0.012 ADOPT**, 0.0077–0.012 LEAN, ≤0.0077 NULL.
POLICY 1 (Log 23:40): final submissions chosen by pooled OOF on grouped folds, never public LB.
POLICY 2 (Log 23:40): no registration-gated external data (MRNet, OAI, …); adoption of any public asset is logged.
**POLICY 5 (Log 08-24 16:20) — GLOBAL DEPLOY FREEZE, IN FORCE**: Felipe's standing instruction "don't deploy". NO Kaggle pushes of any kind (code/kernel/cache/train/ablate/submit, dataset versions, new kernels, metadata flips) and NO competition submissions. Local engineering continues normally. Any step needing a push is PARKED with a note and surfaced to Felipe. **Only Felipe can lift this.** Every schedule item below that requires a click or a push is suspended until he does — the weekly plans stay as written so they can resume unchanged.
**POLICY 3 (Log 08-24 14:35) — spend on GEOMETRY, not CAPACITY**: every capacity/optimisation arm has measured NULL (ViT-B, tb12, ep14, lrb1e4, tb6); both big winners changed where the model looks (coverage +0.0325, reg4 +0.0220). New arms must change the input view, the zoom, the coverage, or the attention geometry. Bigger-encoder-at-224 proposals are closed — do not re-open without new evidence.
**POLICY 4 (Log 08-24 14:35) — pretrained-weight licences**: any non-Apache/permissive weights (DINOv3 is under Meta's dinov3-license) may be MEASURED freely but must clear the competition's external-data/pretrained-weights rules before entering a FINAL submission. Verification is Felipe's (rules page + forum).

## 1. Resource ledger
| resource | budget | basis / rules |
|---|---|---|
| T4 GPU | 30 h/week; **assumed reset Saturday 00:00 UTC** (Log 15:35 counts "week since Sat Aug 22") — **VERIFY-WITH-FELIPE** | ≈21 h committed of the Aug 22–28 week → **≈9 h left now** (Log 15:35, 22:40). Plan ≤24 h/week thereafter; 6 h/week reserve. Sessions ≤8.3 h (TOTAL_MIN 500, Log 15:35) |
| Kaggle GPU sessions | max 2 concurrent (Log 12:05) | never schedule a 3rd; pull remote `enable_gpu` after every push (Log 22:40 rule) — cache kernels flipped to GPU twice (Logs 12:05, 22:40) |
| Submissions | 5/day (UTC; day ends 02:00 CEST, status) | routine ≤2/day; ≥3/day only Oct 17–20; keep best-AUC + efficiency student selected (runbook) |
| Laptop MPS | ≈20 h/day | 224-18img arm ≈6 min/epoch/fold (5.9, Log 17:15; 7.1, Log 12:00) → 2-fold 8-ep arm ≈1.6 h, ~10/day; coverage/336 arms ~3× (17.9 min/ep Log 12:00; 20.1 Log 19:55) → ≈4.8 h, ~4/day |
| Disk | 11 GiB free (status 23:40); floor 9 GiB (min seen 8.9, Log 12:00) | evict caches of settled arms first (P252 precedent, Log 15:10); never evict the current base cache; **never fetch** zoomj 17.7 GB (Log 15:10) or G20 2×11.9 GB (Log 22:40) — T4 kernels attach them; G20 merges in /kaggle/temp (Log 22:40) |
| Clicks | 1 human | batch into two windows: **09:00 and 21:00 CEST** (propose to Felipe); each = "open kernel → Settings → GPU T4 x2 → Save & Run All" per runbook |

## 2. T4 session costs (traceable)
| session type | h | basis |
|---|---|---|
| ablate (3–4 two-fold arms) | ≤8.3 | TOTAL_MIN 500 (Log 15:35) |
| train-final 5-fold ×10 ep, g10-class | ≈6.5 (cap 7.8) | 8-ep 5-fold = 311.8 min (Log 20:28) ×10/8; `--limit-minutes 470` (Log 20:05) |
| SSL (`slotknee-ssl`) | ≈2 | Log 22:40 |
| T4 submit (`slotknee-submit`) | ≈1 | Log 15:35 quota table |
| G20 2-fold gate arm | ≈5 | g10 2-fold = 2.0 h (Log 22:40) ×2 forwards + temp-merge |
| G20 5-fold ×10 ep | ≈13 (2 sessions) | 2× train-final estimate |

## 3. Week-by-week plan
Weeks run Sat–Fri with the assumed quota reset. "Click" = Felipe, at 09:00/21:00 unless stated.

### W0 — Aug 24–28 (≈9 h T4 left) · Phase A
| item | detail |
|---|---|
| T4 A1 (click ~09:00 Mon) | `slotknee-ablate-g10` v1 re-click, 8.3 h — arms zoomj / p336 / g10_tau3 (Log 15:35; v2 cancelled Log 22:55). Answers: zoom gate, resolution-on-coverage, head-fix-on-coverage |
| Laptop | finish queue **relabel** → ep14 → tb12 → lrb1e4 → reg4 → noaux → tokenpool → tb2 → nomixer → tau3 → mil → seqmix → rankloss → g10sub5 (Log 15:35 + 02:05 + 02:30) ≈2.5 days; relabel gate scores vs v4-based y on common uids + gold (Log 02:30 caveat); label quality otherwise CLOSED as a lever (adjudication: 286/300 faithful) — priority axes are vision-side: G20 coverage, zoom views, head arms; then pseudo-labels v2 build (`scripts/pseudo_fill.py`, CPU) |
| Kaggle CPU (API, no click) | g20a/g20b caches finish (Log 22:55); `slotknee-train-final` RECIPE filled from **D1/D2**, pushed, held for W1 click |
| **Fast decode (W0 — CLAIM RETRACTED 15:05)** | decode-bench measured the real data: **100% of visible test AND train files are uncompressed Explicit VR LE** — the 5× premise (borrowed from RSNA mammography) does not apply, and the "decode is 70% of runtime" reading was `info["ms"]` = whole build × 3 TTA shifts. dicomsdl is worth **~5 min (~6.6%)**, not 40; the 0.04–0.06 AUC-equivalents figure is **retired**. Ship it anyway (Kaggle parity: 43/43 study tensors bit-identical, 0 fallbacks). **The real lever is the index-reuse fix** (index once, not once per TTA shift; 1.51× locally, larger on Kaggle cold I/O) — magnitude to be measured in the next real submit run before anything is booked |
| Queue additions at next driver restart | **distill2** (teacher_g10_oof.csv, leak-free) → **aucm** (`--loss aucm --aucm-epochs 3`) → **sam** (rho 0.05, head+4 blocks) — flags must exist in train_slotknee first (plan Log 08-24 12:00) |
| Submissions | read tonight's pair (D0 below); ≤1/day probe only if a gate ADOPTs |
| **D0** (Mon) | sub1 (pure g10 ×2) ≈ sub2 (multi-view ×10+TTA)? If yes → best-AUC entries become pure-g10t1 or w_g10≈0.9 (OOF sweep, Log 23:40); submit kernel gets per-group weight |
| **D1** zoom views (re-ranked 08-24 14:35) | **zoomc (coronal joint line) outranks zoomj**: it targets MCL + both menisci + both OA compartments = 4 of the 6 model-limited findings, vs zoomj's single sagittal ACL notch (whose labels already agree with gold at 0.99). Ablate order once the zoomc cache lands: zoomc → zoomj → g10_tau3 → g10_cutrot → p336 (p336 = drop candidate under quota pressure; already NULL on the laptop). ADOPT → base cache := that zoom layout |
| **D2** head fixes | T4 g10_tau3 + laptop tau3/mil/nomixer: ADOPT-compose; nomixer-vs-tau3/mil conflict → one extra deciding arm (plan step 2) |
| **D3** ep14 (laptop) | ADOPT → train-final epochs := 12–14 (14 ep 5-fold ≈9 h > 470-min cap → 12 ep, or 14 ep split 3+2 folds) |

### W1 — Aug 29–Sep 4 (30 h) · Phase A completion
| item | detail |
|---|---|
| T4 (≈21 h) | B1 `slotknee-train-final` v1: D1/D2/D3 recipe, 5 folds, seed 42 (6.5 h, click Sat 09:00) · B2 same, SK_SEED 1337 (6.5 h, Sun) · B3 `slotknee-ssl` 2 h + B4 G20 2-fold gate arm 5 h (one click window Tue) · B5 T4 submit 1 h (Thu) |
| Laptop | 2-fold confirmations of LEAN arms (2nd seed); pseudo-v2 gate arm (224, 1.6 h); seqmix implementation + smoke (Phase B prep); soup-vs-ensemble pooled-OOF measurement once B1+B2 exist (`scripts/soup_slotknee.py` seeds 42+1337 per fold vs the 2-seed rank average; **efficiency-fallback measurement only** — in-domain ensembles beat soups, research round 2) |
| DINOv3 staging (research round 2) | weights → code dataset: `vits16@224` (efficiency: 196 vs 256 tokens) + `convnext_small.dinov3_lvd1689m` (AUC play @384–448 on ZOOM slots — after zoomj verdict); ToMe inference patch (`--tome-r`, gate ΔOOF ≥ −0.001) for the efficiency entry |
| Submissions (Thu) | best-AUC: train-final 2-seed rank-average (expect OOF ≈0.86+, LB ≈0.875+ via +0.016 offset) |
| **D4** G20 | ADOPT (>2× noise vs g10t1, Log 22:40 gate) → W2 runs 5-fold G20; NULL → coverage saturated, hours go to Phase B |
| **D5** SSL | ADOPT → all later trainings warm-start `encoder.safetensors` (runbook) |
| **D6** pseudo-v2 | ADOPT → labels := v5 blend for every retrain from W2 on |

### W2 — Sep 5–11 (30 h) · Phase A close, Phase B start
- T4: if D4 ADOPT → G20 5-fold ×2 sessions (13 h) then it is the base; else hours go straight to Phase B.
- Deprioritised on evidence (research round 2, plan Log 08-24 12:00): no MIL/attention-aggregation variants beyond the queued mil/seqmix (mean-pool parity benchmark); no big-encoder-at-224 arms beyond the vitb gate (3 independent nulls); soups = efficiency fallback only.
- Phase B arms (definitions + owners dispatched, Log 23:40; each 2-fold, paired on the coverage layout):
  **seqmix** (GRU/ordered transformer along the anchor axis, model owner) · **aclzoom** (per-slot `--zoom-spec
  SAG_FS:80:joint`, data owner; cache CPU-built via API like zoomj, Log 15:10 pattern) · **vitb-on-coverage**
  (weights + code-dataset packaging DONE, Log 23:40; ~2× g10 → **≈5 h/2-fold**, own session). One ablate session
  (8.3 h): seqmix / aclzoom / spare; plus the vitb session (5 h).
- Laptop: seqmix/aclzoom 224-scale pilots first — a laptop NULL kills a T4 arm before it costs hours.
- Submission (Thu): current-best retrain if any gate ADOPTed; else skip (bank the day).

### W3 — Sep 12–18 (30 h) · Phase B measurement
- T4: Phase B ablate session 2 (composals of W2 winners, 8.3 h); mid-campaign 5-fold retrain of best composed recipe, 2 seeds (13 h); T4 submit (1 h).
- **D7**: each Phase B arm gated individually; adopt-compose with the D2 conflict rule.
- Submissions: Mon probe + Thu best-AUC.

### W4 — Sep 19–25 (30 h) · Phase C: self-training iter 1
- Teacher = pooled ensemble OOF (per-fold, leak rule §4) → pseudo-label all 4,407 → retrain 5-fold ×2 seeds (13 h); **D8a**: iterate only on ADOPT.
- `scripts/train_slotknee_pipeline.sh` (auditor-fixed, plan Log 02:15) is a gated candidate driver for these sessions — adopt only if it matches/beats the plain per-fold recipe on pooled OOF.
- Laptop: distill-target build + student pilots (leak rule enforced).

### W5 — Sep 26–Oct 2 (30 h) · Phase C: iter 2 + views×seeds
- Self-train iter 2 if D8a ADOPT (13 h); 3rd seed on best recipe + runner-up view refresh (plan step 4: best AND runner-up view) → target matrix ≈2 views × 3 seeds.
- Weight sweep by pooled OOF (Log 23:40 method); submission Thu of the swept ensemble.

### W6 — Oct 3–9 (30 h) · Phase C: efficiency entry
- Distilled student (`--distill-targets`, plan step 5): full-fit, single ckpt, T4-submit and **time it** — efficiency exchange rate 0.01 AUC ≈ 12 min (docs/research_pivot_efficient_model.md §1).
- Fill remaining seed/view cells; both tracks get a Thu submission.

### W7 — Oct 10–16 (30 h) · Freeze ramp
- **Oct 12: last cache change.** **Oct 15: entry/merge deadline — confirm entry + team state with Felipe (VERIFY).**
- Final full-fit retrains (≥2 seeds, best + runner-up view, plan step 4); daily OOF-selected candidate submissions (≤2/day).
- **Oct 17 (Sat, quota resets): last recipe change.**

### W8 — Oct 17–22 · Final
- Oct 17–19: score final candidates, up to 5 subs/day from banked headroom; both tracks.
- **Oct 20: final 2 selected by pooled OOF** (best-AUC ensemble + distilled student), verified "selected" in the UI at both click windows.
- Oct 21–22: buffer only — resubmission of an already-scored version if Kaggle glitches; no new science.

## 4. Risk register
| risk | mitigation |
|---|---|
| Session restart / cancel (ablate-g10 v2 lost ~3 h, Log 22:55) | everything resumable from disk: per-fold ckpts+OOFs, `--skip`/merge for caches (Log 22:40), driver arm-resume, detached waiters with logged pids (Logs 15:40, 20:05); re-click = resume, arms ordered by priority so partial sessions still answer the top gate |
| GPU quota exhausted | laptop fallback ranks any 224-scale arm (−0.02 offset, plan Fixed facts; rank not level); reserve 6 h/week; drop rule = lowest-priority ablate arm first (Log 15:35 precedent) |
| 2-GPU-session cap blocks pushes/clicks | check running sessions before any push (Log 12:05); retry loops only for the cap error (Log 15:40); verify remote `enable_gpu` after every push (Log 22:40) |
| Public-LB drift / overfit to LB | select by pooled OOF only; LB used as confirmation (~1:1, plan Fixed facts); never chase a LB delta <0.005 (≈390-study public split noise) |
| Teacher leak in distill arms | distill/self-train targets must be strictly per-fold OOF of the teacher (never full-fit preds on training folds); any arm whose targets predate this rule is excluded from gates and rebuilt |
| Fold-split mismatch | every new kernel's first OOF must show uid-ovl 1.0000 vs `baseline_s42` (Log 12:00 fold-fix + verification rule) |
| Single clicker | all clicks batched at 09:00/21:00 CEST (propose to Felipe); stage + diff remote code before the window (Log 16:10 pattern) so each click is <2 min; nothing hard-scheduled needs >2 windows/day |
| Disk | ledger rules §1; check free GiB in every status rewrite; eviction decision logged before deletion (Log 12:00 precedent) |

## 5. Daily loop
**Felipe — 09:00 CEST (≈5 min)**: open the click list → click each listed kernel (T4 x2, Save & Run All) → report the quota widget hours → confirm which 2 submissions show "selected".
**Felipe — 21:00 CEST (≈3 min)**: same table (usually the T4 submit or the 2nd training seed); nothing listed = nothing to do.
**Before each window**: push + pull-diff remote code, verify `enable_gpu` and sources, rewrite the click table with exact slug+version.
**After each window**: confirm RUNNING on T4 (not P100) within 15 min; else re-list for the next window.
**Continuously**: 10-min Kaggle poll; fetch+score every COMPLETE (compare_arms, uid-ovl check); append Log entry per result/decision; submit (≤2/day, only pre-authorized); nightly: pooled-OOF table, quota ledger vs widget, disk GiB, laptop queue health (`logs/ablate.log` staleness).
**Reported daily to main (≤10 lines)**: gates decided, OOF/LB numbers, hours spent/left, next window's clicks.

## Campaign shortlist after the 2026-08-24 pricing round (supersedes earlier W1/W2 rows)

Three levers were priced against the trained champion and CLOSED without a GPU session (see
docs/research_improvements_20260824.md): more anchors (saturated), more resolution (the model
does not use the detail it has), more seeds (+0.0011 at 0.953 correlation). What remains, in
order, is short:

| # | lever | status | needs |
|---|---|---|---|
| 1 | **reg4 5-fold** | measured **+0.0220**, recipe already in `train_kernel_g10`/`train_kernel_final` | ONE T4 session |
| 2 | **zoomc** (coronal joint-line zoom) | the only input lever that CANNOT be priced from the current model — it supplies information the model has never seen | its cache re-built (the build was CANCELLED), then one ablate session |
| 3 | training-time arms (distil / AUCM / SAM) | running on the laptop tonight, 0 s inference cost | nothing |
| — | entry-1 submission | built, ~0.871 expected | Felipe's click |

**Removed from the ablate-g10 queue**: `p336` — the most expensive arm in it (2.25x tokens per
step, most of a session) and now the weakest prior. Its slot goes to the zoom family.

**BLOCKER for #2**: `slotknee-cache-zoomc` shows CANCEL_ACKNOWLEDGED, so the coronal cache does
not exist. Re-running it is a CPU push, which POLICY 5 currently forbids. The ablate kernel
skips an arm whose cache is not attached, so nothing breaks — the arm simply cannot run until
the cache exists. This is the single highest-value item the deploy freeze is holding up, after
the reg4 session.

## ADOPTION STATE, verified against the DEPLOYED kernels (2026-08-25 03:00)

Every measured decision is live on Kaggle, checked by pulling each kernel back and grepping the
deployed source — not by trusting the local tree:

| decision | evidence | where it is live |
|---|---|---|
| **reg4 backbone** | +0.0220 pooled, ACL +0.0617 | `slotknee-train-g10` v11, `slotknee-train-final` v5 |
| **self-distillation** | +0.0185 pooled, every label up (leak caveat: OOF inflated, hidden test unaffected) | both train kernels, teacher ships in the code dataset |
| **fast decoder + index reuse** | pixel-identical, ~78 → ~60 min | `slotknee-submit` v11 |
| **3-shift TTA, no more** | 1→3 shifts +0.0044; 3→5 only +0.0006 | submit kernel, documented in-file |
| **TTA OFF for the efficiency entry** | nets −0.0098 there (buys 0.0044, spends 0.0142) | `SK_TTA=0`, a setting not a code change |
| **seeded submission** | a total failure now scores 0.5, not 0 | submit kernel |
| **zoomc first, p336 dropped** | detail probe: model ignores detail finer than ~112 px | `slotknee-ablate-g10` v7 |

**Measured and deliberately NOT adopted** (all NULL or REGRESS, closed rather than lingering):
MIL pooling −0.0176, sharpness-aware −0.0214, attention temperature −0.0010, AUCM −0.0000,
sequence GRU +0.0041, 12 trainable blocks +0.0054, 14 epochs +0.0061, higher backbone LR
−0.0366, relabelled targets (no-gate), ViT-B (×2), G20 (saturated), 336 px (detail unused),
extra seeds (+0.0011), extra TTA shifts (+0.0006).

**Still deliberately unchanged**: `DEFAULT_BACKBONE` in `src/slotknee.py` stays plain DINOv2 so
in-flight ablation arms remain comparable — flip it at recipe freeze and re-baseline.

**Nothing further can be adopted without GPU hours.** The next adoptable decision is whichever
of `zoomc`, the diversity arms, or the attention family gates — and each needs a session.
