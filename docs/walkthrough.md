# Walkthrough: Pseudo-Labeling Implementation

I have successfully added the Pseudo-Labeling pipeline to your project. This is a massive upgrade that will allow your models to learn from the 4,349 unlabelled studies in the dataset.

## What was changed

### 1. New File: `src/pseudo_label.py`
This script takes the 15 trained checkpoints from Phase 1 and 2, and runs inference on the **unlabelled** rows of `train.csv`. 
- It uses the same Test-Time Augmentation (TTA) as the final inference script to ensure high-quality labels.
- It averages the raw probability outputs across all 15 models (preserving calibration for distillation) and saves them to `data_subset/train_pseudo.csv`.

### 2. Modified: `src/train.py`
The data loader was updated to seamlessly support `--set use_pseudo_labels=true`. 
- When this flag is enabled, it automatically detects `train_pseudo.csv`, loads those predicted labels, and merges them directly into the training dataframe. 
- The `filter_labelled` logic now correctly sees these 4,349 rows as having valid labels and includes them in the training epochs.

### 3. Modified: `scripts/train_gold_pipeline.sh`
The main bash script now has two new phases inserted right before the final submission step:
- **Phase 3**: Executes `src/pseudo_label.py` to generate the predictions.
- **Phase 4**: Fine-tunes Backbone A, Backbone B, and Backbone C for an additional 5 epochs each. Crucially, it uses `--set init_from="..."` to load the best weights from the previous phases, ensuring we are strictly improving the models rather than starting from scratch.

### 4. Uploaded to Kaggle
- A new version (Version 3) of your Kaggle Model `felipedeleon11/nidhogg/pytorch/default` has been successfully pushed.
- The `Kaggle_Gold_Final.zip` on your Desktop has been updated.

## Next Steps
Once your current Kaggle notebook finishes the Phase 1 and 2 training and generates its `submission.csv`, you can either:
1. Submit that initial CSV to the leaderboard to get a baseline score.
2. Re-run the Kaggle notebook! Since the Kaggle Model was updated to Version 3, the next time you run it, it will automatically pull the new script and execute the full Pseudo-Labeling pipeline (Phases 1 through 4).

Let me know if you want to inspect any of the new code or if there is anything else you'd like to work on!
