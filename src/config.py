import dataclasses
import argparse

@dataclasses.dataclass
class Config:
    # Data params
    data_dir: str = "data_subset"
    require_real_data: bool = True
    image_size: int = 224
    samples_per_plane: int = 10
    
    # Training params
    # NOTE: epochs / lr / batch_size are declared ONCE, in the "Optimizer / LR
    # Schedule" block below.  They used to be declared here as well; a dataclass
    # keeps only the LAST definition of a field, so these earlier values were
    # dead and every "tuned" value set here was silently discarded.
    llrd_factor: float = 0.2
    grad_accum_steps: int = 2
    early_stopping_patience: int = 8
    models_dir: str = "models"
    
    # Model params
    backbone: str = "tf_efficientnet_b0_ns"
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 12
    
    # ── Multiple Instance Learning (2.5D MIL) ─────────────────────────────
    # True MRNet style: processes each slice in `in_channels` independently
    # using a 1-channel backbone, then pools features across the slice dimension.
    # Dramatically improves AUC at the cost of processing B*C slices.
    mil_mode: bool = False
    mil_pool: str = "max"               # max | avg | gem
    
    # Advanced configuration
    focal_loss: bool = True           # Focal Loss for class imbalance
    focal_gamma: float = 2.0          # Focal loss gamma
    tta_n: int = 5                    # Number of TTA augmentations at inference
    use_ema: bool = True              # Model Exponential Moving Average
    ema_decay: float = 0.999          # Decay rate for EMA
    gradient_checkpointing: bool = True  # Recompute activations to save VRAM

    # ── Stochastic Weight Averaging ───────────────────────────────────────
    # The single cheapest generalisation gain available here: average the
    # weights of the post-decay epochs into one model.  Costs one fp32 CPU
    # copy (no VRAM, no extra inference), and train.py only keeps the average
    # if it beats the val-selected checkpoint on the held-out fold, so it can
    # never make a fold worse.
    use_swa: bool = True
    swa_start_frac: float = 0.6       # start averaging after 60% of the schedule
    use_lookahead: bool = True        # Lookahead optimizer wrapper (k=6, alpha=0.5)
    val_tta: bool = True              # orig + laterality-corrected hflip at val time
    
    # Other
    use_pseudo_labels: bool = False
    smoke_test: bool = False

    # ── Label handling ────────────────────────────────────────────────────
    # data_subset/train.csv has 4407 rows but only 58 carry labels; the other
    # 4349 are NaN across all 12 targets. RSNADataset does nan_to_num(labels, 0)
    # so those rows arrive at the loss as *confident all-negative* studies.
    # train_csv points at the pre-stratified 58-row file; labelled_only is the
    # belt-and-braces filter that also protects the plain train.csv path.
    train_csv: str = "train_gold.csv"   # relative to data_dir; falls back to train.csv
    labelled_only: bool = True          # drop rows with any NaN target
    n_folds: int = 5

    # ── Pseudo-labelling / self-training ──────────────────────────────────
    # See src/pseudo_label.py for the full rationale.  Summary:
    #   * Pseudo-labels MUST be out-of-fold.  A study's pseudo-label may only
    #     come from teacher models that did NOT train on the gold fold the
    #     student will validate on, otherwise gold validation labels leak into
    #     the student's training targets and val AUC rises without any real
    #     generalisation gain.  pseudo_label.py therefore writes ONE ROW PER
    #     (study, teacher_fold) and train.py picks the matching slice per fold.
    #   * Pseudo rows are NEVER validated on.  Validation is gold-only.
    #   * The 58-study teacher is weak, so its guesses are filtered by
    #     confidence and down-weighted relative to gold.
    pseudo_csv: str = "train_pseudo.csv"      # relative to data_dir
    # Keep a pseudo-label cell only if |p - 0.5| >= this.  0.30 => p<=0.20 or
    # p>=0.80.  With a 58-sample teacher, anything closer to 0.5 than this is
    # indistinguishable from the prior and is pure confirmation-bias fuel.
    pseudo_conf_margin: float = 0.30
    # Optional hard cap: per (teacher_fold, label), keep at most this many of
    # the most confident positives and this many of the most confident
    # negatives.  0 = no cap (margin filter only).
    pseudo_max_per_label: int = 0
    # Loss weight of a surviving pseudo cell.  Gold cells are always 1.0.
    pseudo_weight: float = 0.30
    # Refuse to run if out-of-fold provenance cannot be proven (recommended).
    pseudo_strict_oof: bool = True
    # Belt-and-braces: never let a pseudo row into the validation split.
    pseudo_validate_on_gold_only: bool = True

    # ── Weak supervision from radiology reports ───────────────────────────
    # See src/labels.py for the rules and src/kaggle_data.py's
    # attach_report_label_rows() for the single integration hook.
    #
    # The opportunity: data_subset/train.csv has 4407 rows and EVERY one has a
    # non-empty `Report`.  649 of those studies have pixels on disk; only 58
    # have expert labels.  Report-derived soft targets turn 58 trainable
    # studies into 649 at zero inference cost, and they are INDEPENDENT
    # evidence -- a radiologist looked at the knee and wrote what they saw --
    # unlike pseudo-labels, whose teacher is the same weak 58-study model.
    #
    # Measured against the 58 gold studies: macro ROC-AUC 0.869 in-sample /
    # 0.815 leave-one-fold-out, per-verdict reliability
    #   rule says positive -> 78% correct | negative -> 97% correct
    #   unmentioned        -> 12.6% still positive, hence a PRIOR, never a 0.
    #
    # DEFAULT OFF so no other run changes behaviour unexpectedly.
    use_report_labels: bool = False
    report_csv: str = "train.csv"       # relative to data_dir; must have `Report`
    report_text_col: str = "Report"
    # Loss weight of a report-derived cell at full rule confidence.  Gold cells
    # are always 1.0 and are never overwritten (report studies are new rows).
    report_label_weight: float = 0.35
    # Drop a report cell whose rule confidence is below this.  An "unmentioned"
    # cell reports confidence 0.0 -- its TARGET is the calibrated prior (never
    # 0, which would fabricate a negative) but it costs the loss nothing.  A
    # hedged finding lands near 0.33 and survives this gate, proportionally
    # down-weighted; raise to 0.9 to keep only unhedged calls.
    report_min_confidence: float = 0.10
    report_labels_require_images: bool = True   # a report with no pixels is not a sample
    report_label_max_studies: int = 0           # 0 = no cap (debugging aid)
    # Optional: refit the label calibration on these gold folds only, so the
    # constants carry no trace of the fold being validated.  Empty = use the
    # constants shipped in src/labels.py.
    report_label_calibrate_on_folds: str = ""

    # ── Loss selection (A/B-able) ─────────────────────────────────────────
    # "bce" | "focal" | "asl" | "auto" ("auto" defers to the legacy focal_loss flag)
    # ASL is the recommended default for multi-label imbalanced problems.
    loss: str = "asl"
    focal_alpha: float = 0.25           # < 0 disables alpha weighting
    asl_gamma_neg: float = 4.0          # ASL: focus parameter for negatives (paper default)
    asl_gamma_pos: float = 1.0          # ASL: focus parameter for positives
    asl_clip: float = 0.05             # ASL: probability margin for negatives
    ohem_ratio: float = 0.70            # OHEM: backprop only the hardest 70% of pixels in the batch

    # ── Model selection / early stopping ──────────────────────────────────
    # "auto" | "val_loss" | "val_auc" | "val_crit"
    #
    # WHY THE DEFAULT IS NOT val_auc.  Macro ROC-AUC is a RANK statistic: with p
    # positives and n negatives it can only move in steps of 1/(p*n).  Each fold
    # here validates on 10-13 studies (12/10/12/11/13), so the macro step is
    # 0.0027-0.0041.  A run in this project was killed by early stopping on a
    # 0.0003 "non-improvement" -- 9-14x below the finest change the metric can
    # express -- while train and val loss were still falling monotonically.
    #
    # THE DEFAULT IS val_loss: masked BCE over EVERY validation cell (~130-155
    # supervised cells per fold vs 12 AUC numbers), continuous, defined for
    # every study, and monotone with genuine improvement.  AUC is still computed
    # and logged every epoch -- it just does not drive control flow.
    #
    # "auto" is the opt-in forward path for when the panel grows: it measures
    # the fold's own AUC quantisation and uses val_auc only when BOTH
    #   * one rank swap moves macro AUC by <= auc_resolution_target, AND
    #   * the fold validates on >= auc_min_val_rows studies -- because with 12
    #     studies the SAMPLING error of an AUC (~0.1) dwarfs its quantisation,
    #     so passing the resolution test alone is not evidence of anything.
    # At 58 gold studies "auto" resolves to val_loss (10-13 rows per fold);
    # once src/labels.py grows the panel to 649 it resolves to val_auc by
    # itself, with no code change and no hard-coded study count.
    selection_metric: str = "val_loss"
    min_epochs: int = 3                 # never early-stop before this many epochs

    # ── Selection resolution (min_delta) ──────────────────────────────────
    # Changes smaller than the metric's own resolution are classified
    # "within_resolution": the incumbent checkpoint is kept AND early-stopping
    # patience is NOT charged.  Ties being patience-neutral means a LARGER
    # min_delta can only ever delay stopping, never hasten it.
    #
    # < 0 (default) = derive it from the data:
    #   * rank metrics (val_auc): mean_c[1/(p_c*n_c)] / K over the K computable
    #     labels of THIS fold -- one rank swap, macro-averaged.
    #   * loss metrics (val_loss/val_crit): the standard error of the PAIRED
    #     per-study loss difference against the incumbent epoch, times
    #     selection_loss_delta_k.  Paired because the validation set is
    #     identical between epochs, so the study-to-study spread cancels.
    # 0.0 reproduces the old strict-inequality comparison exactly.
    selection_min_delta: float = -1.0
    selection_min_delta_reduce: str = "mean"   # "mean" | "min" over per-label steps
    selection_loss_delta_k: float = 1.0        # multiples of the paired SE
    # "auto" prefers val_auc once one rank swap moves macro AUC by <= this
    # AND the fold has at least auc_min_val_rows validation studies.
    auc_resolution_target: float = 0.005
    auc_min_val_rows: int = 100
    # Stop after this many consecutive epochs with no measurable progress
    # (ties + regressions).  0 disables; None-equivalent default is
    # max(2*early_stopping_patience, early_stopping_patience+2).
    stale_patience: int = 0
    # Reproduce the pre-fix selection arithmetic bit for bit (min_delta=0, no
    # NaN fallback, ties charged to patience).  Only for reproducing old runs.
    selection_legacy: bool = False

    # ── Validation knobs (read via getattr in train.py; declared so --set works)
    # (val_tta is declared once, in the "Advanced configuration" block above.)
    val_ema_alpha: float = 0.4          # smoothing of the logged score EMA
    # Validation is computed on GOLD ROWS ONLY, always.  Report-derived or
    # pseudo rows measure agreement with a keyword matcher, not with ground
    # truth, so they must never enter a validation split.
    val_gold_only: bool = True
    gold_column: str = "is_gold"

    # ── Report-derived (weak) training labels ─────────────────────────────
    # Produced by src/labels.py (a separate module).  Purely additive:
    # with labels_csv empty, or the file absent, nothing changes.
    # Derived rows are appended to the TRAINING frame with fold = -1, so they
    # can never reach a validation split, and carry a lower loss weight.
    labels_csv: str = ""                # e.g. "train_derived.csv", rel. to data_dir
    labels_weight: float = 0.30         # loss weight of a derived cell
    gold_weight: float = 1.0            # loss weight of a gold cell
    labels_min_conf: float = 0.0        # drop derived cells below this |p-0.5|*2
    soft_targets: bool = True           # accept targets anywhere in [0, 1]
    # Column in labels_csv holding a per-row confidence in [0,1]; multiplied
    # into labels_weight when present.  Empty = ignore.
    labels_weight_column: str = "sample_weight"
    # Per-CELL confidence columns, one per target: src/labels.py emits
    # "<label>__conf" next to "<label>".  An "unmentioned" finding comes back
    # as (prob = class prior, confidence = 0.0), so multiplying the confidence
    # into the weight makes it cost exactly nothing -- no threshold needed.
    # Empty string disables the lookup and falls back to |p - 0.5| * 2.
    labels_conf_suffix: str = "__conf"

    # Label smoothing applied INSIDE the loss.  train_one_epoch already smooths
    # the targets, so this stays 0 to avoid smoothing twice (which pushed ASL's
    # targets to 0.025/0.975 and made every cell read as positive).
    loss_label_smoothing: float = 0.0

    # ── Wall-clock guard ──────────────────────────────────────────────────
    # "wall"      : max(time.time(), time.monotonic()) elapsed -- counts machine
    #               suspend, which is what Kaggle bills.  Correct on Kaggle.
    # "monotonic" : time.monotonic() only -- excludes suspend, so a laptop that
    #               sleeps mid-run does not lose its budget.  Local dev only.
    time_budget_clock: str = "wall"

    # ── Progressive-resolution phases / resumability ──────────────────────
    init_from: str = ""                 # weights-only warm start (previous phase)
    resume: bool = True                 # auto-resume from fold_{f}_last.pt
    time_budget_min: float = 0.0        # 0 = no wall-clock guard
    state_dir: str = ""                 # defaults to models_dir

    # ── Runtime ───────────────────────────────────────────────────────────
    amp_dtype: str = "auto"             # auto | bf16 | fp16 | fp32
    num_workers: int = 4

    # ── Optimizer / LR Schedule ───────────────────────────────────────────
    lr: float = 3e-4
    epochs: int = 12
    batch_size: int = 8
    warmup_epochs: float = 1.0          # warmup for OneCycleLR/Cosine
    # Reaches the optimiser for real now.  _make_param_groups used to hard-code
    # 1e-4 and ignore this field, so every "heavy weight decay" run was actually
    # a 1e-4 run.  Norms and biases are still excluded from decay.
    weight_decay: float = 0.05

    # ── GPU utilisation (owned by src/train.py) ───────────────────────────
    # ADDITIVE BLOCK -- every field below is read through getattr() in
    # train.py, so an older Config still runs.
    #
    # THE PROBLEM THESE SOLVE.  batch_size=8 at 224px on effnet_b0 costs
    #   0.97 GiB fixed (params+grads+AdamW+workspace) + 8 x 43.1 MiB activations
    #   = 1.30 GiB = 8.7% of a T4's 15.0 GiB usable.
    # MEASURED on this machine (scripts/bench_gpu_util.py, MPS, r2=1.0000):
    #   t_update = 25.7 ms fixed  +  8.8 ms per micro-batch  +  4.60 ms per sample
    # so at batch 8 / accum 2 the fixed terms are 43/117 ms = 37% of wall clock.
    # Batch is a THROUGHPUT knob here, not a memory knob.
    #
    # ...but the dataset caps it.  With 58 gold studies a fold trains on 46, so
    # "batch 128" is one full-batch step per epoch: no gradient noise, no BN
    # statistics worth the name, and only `epochs` optimiser updates in total.
    # auto_batch_size therefore takes the MINIMUM of a memory cap and a data
    # cap, and the data cap is what binds in the gold-only regime.
    auto_batch_size: bool = True
    # Minimum optimiser updates per epoch the data cap must preserve.  This is
    # the ONE knob that decides how much of the card gets used, so it is worth
    # understanding.  batch = n_train // auto_batch_target_steps, floored at
    # auto_batch_min and capped by auto_batch_max and VRAM.
    #
    #   58 gold studies  (46 train): 46 // 16 = 2 -> floored to 8.  The 58-study
    #       regime is ALREADY correctly sized at batch 8; nothing here changes
    #       it, because a bigger batch would be most of the fold.
    #   649 weak studies (519 train): 519 // 16 = 32 -> batch 32, 16 updates per
    #       epoch, 2.31 GiB (15.4% of a T4).
    #
    # 16, not 8, on purpose.  Batch 64 (target_steps=8) is only ~9% faster than
    # batch 32 but HALVES the optimiser updates, and on 519 studies with 12
    # epochs the update count is the scarce resource, not the FLOPs.  Lower this
    # to 8 (batch 64) or 4 (batch 128) only if the wall-clock budget is binding.
    auto_batch_target_steps: int = 16
    auto_batch_min: int = 8
    auto_batch_max: int = 128           # past ~128 the throughput curve is flat
    auto_batch_headroom: float = 0.85   # fraction of card the plan may occupy
    auto_batch_device: str = "t4"       # key into model.DEVICE_VRAM_GIB
    # Accumulation exists to SIMULATE a big batch when memory is short.  With
    # 13.7 GiB idle it buys nothing and costs one extra fixed 8.8 ms block per
    # micro-batch, so auto sizing folds it into batch_size and sets this to 1.
    auto_grad_accum: bool = True

    # Gradient checkpointing recomputes activations instead of storing them:
    # MEASURED 4.70 vs 43.10 MiB/sample for effnet_b0@224 (9.2x less) for a
    # MEASURED +28-51% wall-clock cost at batch 64 on MPS.  Trading 0.30 GiB of
    # a 15.0 GiB card for a third of the run time is a bad trade, so train.py
    # turns it OFF whenever the plan already fits.  Set False to keep whatever
    # `gradient_checkpointing` / `grad_checkpointing` say.
    auto_grad_checkpointing: bool = True

    # ── LR scaling with effective batch (owned by src/train.py) ───────────
    # Raising the effective batch without raising the LR underfits: it is the
    # same number of epochs with 1/k the optimiser updates.
    #   "sqrt"   lr *= sqrt(eff_batch / lr_base_batch)   <-- DEFAULT
    #   "linear" lr *=      eff_batch / lr_base_batch    (Goyal et al. 2017)
    #   "none"   leave cfg.lr alone
    # WHY SQRT AND NOT LINEAR.  Linear scaling was derived for SGD+momentum,
    # where the update is proportional to the raw gradient, so halving gradient
    # variance really does license twice the step.  AdamW normalises by the
    # second moment, so its step size is already scale-free and the linear rule
    # over-corrects.  On top of that, this run has 12 epochs over 46-519
    # studies: 8x the batch means 8x FEWER updates (72 -> 9 in the gold regime),
    # and an 8x LR on a 4M-parameter pretrained backbone with single-digit
    # update counts destroys the pretrained weights before warmup ends.
    # sqrt(8) = 2.83x recovers most of the lost step count at a fraction of
    # the blow-up risk.  Set "linear" only in the weak-label regime AND with a
    # longer schedule.
    lr_scale_rule: str = "sqrt"         # sqrt | linear | none
    # Effective batch cfg.lr was tuned at.  16 = the shipped 8 x 2.
    lr_base_batch: int = 16
    lr_scale_max: float = 4.0           # hard ceiling on the multiplier
    # Absolute floor on warmup, in optimiser updates.  pct_start is a FRACTION,
    # so a bigger batch silently shortens warmup in wall-clock terms exactly
    # when a bigger batch needs more of it.  warmup_epochs (which nothing read
    # before) now drives pct_start, and this is its floor.
    warmup_min_steps: int = 3

    # ── Regularisation (Anti-Overfitting) ─────────────────────────────────
    # With 58 labelled studies these are load-bearing, not garnish.
    drop_path_rate: float = 0.20        # Stochastic Depth for ViT/ConvNeXt
    n_dropout: int = 5                  # Multi-Sample Dropout passes in the classification head
    mixup_alpha: float = 0.4            # 0.0 = disable mixup/cutmix
    label_smoothing: float = 0.05       # Soften hard 0/1 labels to prevent overconfidence
    cache_dir: str = "/tmp/rsna_cache"  # resolution suffix appended automatically
    seed: int = 42

    # ── Augmentation (owned by src/kaggle_data.py) ────────────────────────
    # All scalars, so Config.from_args' `orig_type(v)` conversion works with
    # `--set aug_hflip_p=0.5`. Every knob is read via getattr() in
    # kaggle_data.py, so the dataset still runs against an older Config.
    aug_enabled: bool = True
    # Draw ONE set of transform params per study and apply it to the whole
    # slice stack, so the 2.5D channels stay spatially registered.
    aug_slice_consistent: bool = True

    # ── Slice selection (owned by src/kaggle_data.py) ─────────────────────
    # Set all four legacy values below to reproduce the pre-fix tensors
    # bit-exactly:  slice_order=filename  slice_selection=study_pooled
    #               slice_trim_frac=0.15  slice_sample=endpoints
    #
    # "geometric" sorts each series by ImagePositionPatient projected on the
    # slice normal (InstanceNumber as fallback).  "filename" is the legacy
    # lexicographic sort of SOP-Instance-UID filenames, which measurement shows
    # is uncorrelated with anatomy (|Spearman rho| = 0.15 over 212 real series;
    # 0/212 above 0.9).  Do not use "filename" for anything but A/B tests.
    slice_order: str = "geometric"
    # "plane_balanced" | "single_series" | "study_pooled" (legacy).
    # plane_balanced gives every channel a fixed anatomical meaning; the legacy
    # study_pooled mode drew channels from >1 plane in 95.8% of studies with the
    # plane->channel mapping decided by UID hash.
    slice_selection: str = "plane_balanced"
    slice_plane_priority: str = "Sagittal,Coronal,Axial"
    # Fraction of each series dropped from EACH end before sampling.
    # 0.0, not the old 0.15: on 178 geometrically-sorted series, 89.7% of the
    # slices a 15% trim discards on Sagittal (93.9% on Axial) carry >= half the
    # series' peak tissue area — the same as the slices it keeps. On Sagittal
    # those 13.2 mm per side are the medial and lateral compartments.
    slice_trim_frac: float = 0.0
    # "bin_center" (split the range into k bins, take each centre — spreads over
    # the whole run without ever selecting the outermost, genuinely-empty slice)
    # or "endpoints" (legacy np.linspace including both extremes).
    slice_sample: str = "bin_center"
    # Cache the per-study geometric ordering as a JSON sidecar so the header
    # pass is paid once per study for the whole run, not once per resolution.
    slice_index_cache: bool = True
    # Pixel cache. Turning it OFF is the only way aug_slice_jitter_frac can
    # actually vary the slice window per epoch (a cached study is frozen).
    aug_slice_jitter_frac: float = 0.10
    cache_slices: bool = True
    # Never memoise an all-black stack: a transient read error would otherwise
    # be frozen into the cache and served as a knee for the rest of the run.
    cache_reject_zero: bool = True

    # ── Cache placement / disk budget (owned by src/kaggle_data.py) ────────
    # At 4,407 studies the pixel cache is 0.66 GiB (in_channels=3, 224px) to
    # 5.85 GiB (in_channels=9, 384px) PER RESOLUTION, plus ~0.13-0.21 GiB of
    # JSON slice-index sidecars — see kaggle_data.estimate_cache_bytes(), whose
    # pixel term is exact. Two Kaggle placements lose the session:
    #   * /kaggle/working is the notebook OUTPUT directory (~19.5 GiB limit,
    #     shared with the checkpoints, and 8,814 extra files to commit);
    #   * any tmpfs is RAM-backed, i.e. charged to the same 13-16 GiB the
    #     training process needs.
    # cache_auto_locate lets kaggle_data.resolve_cache_dir() measure free space
    # and filesystem type at runtime and pick a root that provably fits.
    cache_auto_locate: bool = True
    # Free space to leave untouched. Pre-caching stops cleanly when it would
    # eat into this rather than filling the disk mid-run.
    cache_reserve_gb: float = 1.0
    # Hard ceiling on cache growth; 0 = only the reserve applies.
    cache_max_gb: float = 0.0
    # True turns the pre-cache budget shortfall from a warning into a refusal
    # to start. Off by default: the estimate assumes every study yields a cache
    # file, so it over-states when studies are missing from the mount.
    cache_require_space: bool = False

    # Laterality-mirroring flip. Default OFF: mirroring a coronal/axial knee
    # swaps medial<->lateral, which invalidates Medial/Lateral Meniscus,
    # Medial/Lateral OA and MCL unless the labels are swapped with the pixels.
    aug_hflip_p: float = 0.0
    aug_hflip_require_label_swap: bool = True   # refuse to mirror without swap
    # Superior<->inferior flip. Default OFF: a knee is never imaged upside down.
    aug_vflip_p: float = 0.0

    # Small rigid/affine jitter — patient positioning varies between scans.
    aug_affine_p: float = 0.7
    aug_shift_limit: float = 0.0625
    aug_scale_limit: float = 0.10
    aug_rotate_limit: float = 12.0

    # Mild non-rigid deformation (elastic OR grid distortion, never both).
    aug_deform_p: float = 0.20
    aug_elastic_alpha: float = 40.0
    aug_elastic_sigma: float = 6.0
    aug_grid_num_steps: int = 4
    aug_grid_distort_limit: float = 0.05

    # Occlusion. Hole size is a FRACTION of the image side, kept small so a
    # hole cannot swallow the structure that carries the label.
    aug_dropout_p: float = 0.25
    aug_dropout_max_holes: int = 3
    aug_dropout_hole_frac_min: float = 0.03
    aug_dropout_hole_frac_max: float = 0.06

    # ── Inference cost (src/infer.py only; nothing here touches training) ──
    #
    # `batch_size` above is a TRAINING batch size: it is sized against stored
    # activations plus optimiser state. Inference keeps neither, so reusing it
    # leaves the accelerator almost idle. These knobs let src/infer.py size the
    # forward pass against the card that is actually present.
    #
    # infer_batch_images: IMAGES per forward call, counting TTA views
    #   (a batch of 16 studies x 5 views = 80 images). 0 = measure
    #   bytes/image on the real device and fill infer_vram_fraction of it.
    infer_batch_images: int = 0
    # Hard cap. MEASURED (effnet_b0 @224, 5 views): throughput plateaus once a
    # forward carries ~40-80 images and then degrades sharply when the batch
    # stops fitting comfortably -- 320 images was 50x SLOWER per study than 80
    # on a unified-memory device. 256 sits well past saturation with room to
    # spare; the VRAM probe lowers it further whenever the card says so.
    infer_max_batch_images: int = 256
    infer_vram_fraction: float = 0.60   # of FREE VRAM; the rest absorbs
                                        # fragmentation and cuDNN workspaces
    # Collated CPU batches held between models so the second and later models of
    # a preprocessing group skip the .npy read + transform entirely. Bounded, and
    # simply not used when the test set is larger than the budget.
    infer_ram_cache_gb: float = 4.0
    infer_compile: bool = True          # torch.compile warm-up is ~1 min/model;
                                        # worth it only for a large test set
    # Fraction of studies routed to the FULL ensemble; the rest are decided by
    # model 1 alone (routing signal: spread across TTA views). 0 = off, i.e.
    # every model sees every study. Raise this only with harness evidence from
    # scripts/benchmark_inference.py.
    cascade_frac: float = 0.0

    # Intensity — scanner/protocol variation.
    aug_brightness_contrast_p: float = 0.5
    aug_brightness_limit: float = 0.20
    aug_contrast_limit: float = 0.20
    aug_gamma_p: float = 0.20
    aug_bias_field_p: float = 0.25
    aug_bias_field_coeff: float = 0.35
    aug_bias_field_order: int = 3
    aug_clahe_p: float = 0.30
    aug_clahe_clip: float = 2.0
    aug_sharpen_p: float = 0.20
    aug_noise_p: float = 0.20
    aug_noise_std_min: float = 0.01     # fraction of full scale (~2.5/255)
    aug_noise_std_max: float = 0.05     # fraction of full scale (~13/255)

    # Normalisation (single-channel ImageNet grey stats, broadcast over slices).
    norm_mean: float = 0.485
    norm_std: float = 0.229

    # ── Backbone selection (owned by src/model.py) ────────────────────────
    # `backbone` above accepts a short alias from model.BACKBONE_ALIASES
    # (effnet_b0 | effnetv2_s | effnetv2_m | convnext_base | dinov2_small |
    #  dinov2_base | dinov2_large) or any raw timm model name.
    #
    # Kaggle code competitions are OFFLINE: timm cannot download weights at
    # runtime. Attach them as a Kaggle Dataset and point backbone_weights at
    # the file (or the directory, if it holds exactly one checkpoint), e.g.
    #   backbone_weights=/kaggle/input/timm-dinov2-large/model.safetensors
    # Empty string = normal (online) timm download.
    backbone_weights: str = ""

    # DINOv2 is patch-14, so its input side must be a multiple of 14
    # (224 / 322 / 378 / 392 / 448 / 518). model.valid_image_size(backbone,
    # image_size) returns the size that will actually be used; anything else is
    # snapped and the input is resized inside the model.
    #
    # ViT token pooling: "cls" | "avg" | "cls_avg". "cls_avg" concatenates the
    # CLS token with the mean patch token (the DINOv2 linear-probe recipe) and
    # therefore doubles the head's input width.
    vit_pool: str = "cls_avg"
    # CNN spatial pooling: "gem" | "avg" | "avgmax" ("avgmax" doubles width).
    cnn_pool: str = "gem"

    # (drop_path_rate is declared once, above, at 0.20.  A second declaration
    #  here at 0.0 used to override it, so stochastic depth was always OFF.)
    # Recompute activations in the backward pass: ~35% slower, but cuts
    # activation memory several-fold. Needed for DINOv2-L at 518px on a T4.
    grad_checkpointing: bool = False

    # ── Capacity control (owned by src/model.py) ──────────────────────────
    # NOTE: Config.from_args silently ignores `--set k=v` for any k that is not
    # a field here, so these must exist for the knobs to be reachable from the
    # command line at all.
    #
    # With only ~58 labelled studies, how much of the backbone is TRAINABLE
    # matters more than which backbone it is. MEASURED trainable-parameter
    # counts for effnet_b0 (4.03M total): freeze_blocks=0 -> 4.03M,
    # 3 -> 3.96M (98.4%! EfficientNet is back-loaded, a shallow freeze is
    # nearly a no-op), 6 -> 1.15M (28.5%), 7 -> 0.43M (10.7%),
    # freeze_backbone=True -> 17.9k (0.4%, a linear probe).
    # A ViT freezes linearly instead: each dinov2_small block is ~8.2%.
    freeze_blocks: int = 0            # freeze stem + first N backbone stages
    freeze_backbone: bool = False     # freeze the whole backbone (linear probe)
    # Keep frozen BatchNorms in eval mode. requires_grad=False does NOT stop
    # running_mean/var from being overwritten by every training forward pass,
    # so without this a "frozen" stem still drifts towards 58 studies.
    freeze_norm_stats: bool = True

    # Base rate for the head's multi-sample dropout: pass i uses
    # head_dropout * (i + 1), so 0.1 with n_dropout=5 gives 0.1 .. 0.5.
    head_dropout: float = 0.1
    head_norm: bool = True            # LayerNorm on the pooled feature vector

    # Patch-14 ViTs: raise instead of silently snapping 384 -> 378.
    # timm does NOT raise on a non-multiple img_size -- it builds a 27x27 grid
    # and the stride-14 conv discards the trailing 6px of every image (verified:
    # perturbing only rows 378..383 changes the logits by exactly 0.0).
    strict_image_size: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls()
        if hasattr(args, "set") and args.set:
            for s in args.set:
                if "=" in s:
                    k, v = s.split("=", 1)
                    if not hasattr(cfg, k):
                        continue
                    
                    # Type conversion
                    orig_type = type(getattr(cfg, k))
                    if orig_type == bool:
                        setattr(cfg, k, v.lower() in ("true", "1", "yes"))
                    else:
                        setattr(cfg, k, orig_type(v))
        return cfg
