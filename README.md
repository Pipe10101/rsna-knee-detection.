# SlotKnee-S: RSNA Knee Abnormality Detection

This repository contains my ultimate source code for the **RSNA Knee Abnormality Detection** Kaggle competition. 

I designed it to win both the **Main Track (Accuracy)** and the **Efficiency Track** through a unique pipeline I built, known as **Cross-Architecture Knowledge Distillation**.

## 🧠 The Architecture

My pipeline processes multi-sequence 3D MRIs by mapping them into 6 semantic "slots" and extracting exactly 10 spatial 2D anchors per sequence (`g10t1` Coverage Layout). I leverage a custom "Slot Attention" head on top of the `DINOv2` (with Registers) foundation model.

For a full, deep-dive into my architectural mechanics, training passes, and technical rationale, please read my [Architecture Guide](docs/architecture_guide.md).

## 🚀 Execution & Reproduction

My training pipeline uses a 3-stage Teacher-Student loop. All of the core execution logic is contained within my `scripts/` directory.

My primary driver script is:
```bash
scripts/train_slotknee_pipeline.sh
```

### Pipeline Overview:
1. **Pass 1:** I train a `ViT-Base` (reg4) Teacher on 4,407 noisy LLM-extracted labels to find high-confidence predictions for "silent" cells.
2. **Pass 2:** I train a `ViT-Base` (reg4) 5-Fold Ensemble on the resulting 100% dense dataset to maximize raw accuracy.
3. **Pass 3:** I have the massive `ViT-Base` ensemble distill its intelligence (Out-Of-Fold probabilities) into a tiny `ViT-Small` (reg4) student model, securing blazing-fast inference speeds.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
