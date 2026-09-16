# Implementation Plan: Pseudo-Labeling

Pseudo-labeling is the final step to maximize AUC. It involves taking the 15 models we just trained, using them to predict the labels for the 4,349 unlabelled studies in `train.csv`, and then fine-tuning the models on this new, massive combined dataset.

## Proposed Changes

### 1. New Script: `src/pseudo_label.py`
Create a new script (or modify `src/infer.py`) specifically designed to run inference on the **unlabelled** rows of `train.csv`.
*   It will load all 15 checkpoints.
*   It will predict the probabilities for the 4,349 unlabelled studies using TTA.
*   It will save the results to `data_subset/train_pseudo.csv`.

### 2. Modify: `src/train.py` & `src/kaggle_data.py`
Update the data loading pipeline to support `cfg.use_pseudo_labels`:
*   In `load_dataframe()`, if `cfg.use_pseudo_labels` is true, load `train_pseudo.csv`.
*   Merge the pseudo-labels into `train.csv`, filling the `NaN` values for the unlabelled rows with the model's soft predictions (probabilities).
*   Ensure `filter_labelled` does not drop these rows when pseudo-labels are active.

### 3. Modify: `scripts/train_gold_pipeline.sh`
Add Phase 3 and Phase 4 to the bash script:
*   **Phase 3:** Execute `src/pseudo_label.py` to generate `train_pseudo.csv` using the 15 trained models.
*   **Phase 4 (Pseudo-Training):** Re-run training for Backbones A, B, and C for 5-10 additional epochs, passing the flag `--set use_pseudo_labels=true`. This fine-tunes the models on all 4,407 studies.
*   Update the Kaggle zip (`Kaggle_Gold_Final.zip` and the Kaggle API push) to include these new changes.

## Open Questions for You
*   For Phase 4 (Retraining with Pseudo-Labels), do you want to train from scratch on the combined dataset, or just fine-tune the existing best checkpoints for a few extra epochs? (Fine-tuning is much faster, training from scratch is sometimes slightly more robust). I recommend **fine-tuning**.

## Verification Plan
*   Run the pipeline locally with a small dummy dataset to ensure `train_pseudo.csv` is correctly generated and successfully loaded during the retraining phase without crashing.
