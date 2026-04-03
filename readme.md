# AquaFastNet

**AquaFastNet: Underwater Vision Enhancement for Robotics**

AquaFastNet is a lightweight deep learning framework for underwater image enhancement, designed for robotic perception, embedded deployment, and reproducible experimentation. The project focuses on fast inference, stable training, and practical evaluation on paired underwater image datasets.

---

## Overview

Underwater imagery often suffers from reduced visibility, low contrast, color distortion, and detail loss due to absorption and scattering. AquaFastNet addresses these issues with an efficient encoder-decoder architecture that enhances underwater images while remaining lightweight enough for robotics and edge-oriented workflows.

This repository contains the model, training pipeline, evaluation scripts, inference utilities, manifest generation tools, and experiment files used in this work.

---

## Key Features

- Lightweight U-Net style architecture
- Depthwise separable convolutions for computational efficiency
- Squeeze-and-Excitation attention blocks
- Residual enhancement output
- Combined reconstruction and perceptual training objective
- Mixed precision training support
- Exponential Moving Average for stabilization
- Checkpoint resume support
- Video and folder inference utilities
- Evaluation and comparison scripts
- SLURM scripts for cluster training and benchmarking

---

## Repository Structure

```text
AquaFastNet/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements_locked.txt
├── train.py
├── live_video.py
├── compare_visuals.py
├── eval_metrics.py
├── eval_all_metrics.py
├── export_torchscript.py
├── manifests/
│   ├── bench_euvp_test515.csv
│   ├── euvp_paired_train.csv
│   ├── teacher_pseudo_pairs.csv
│   ├── test_pairs.csv
│   ├── train_pairs_train.csv
│   ├── train_pairs_val.csv
│   └── uieb_pairs.csv
├── scripts/
│   ├── build_manifests.py
│   ├── clean_teacherA.py
│   ├── collab_style_test_oncustom_images.py
│   ├── compare_side_by_side.py
│   ├── count_everything.sh
│   ├── enhance_video.py
│   ├── eval_benchmark.py
│   ├── infer_folder.py
│   ├── split_manifest.py
│   └── validate_like_train.py
├── uie/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── models.py
│   └── models1.py
└── runs/
    ├── uie_fastunet_base32/
    │   ├── best.pt
    │   └── last.pt
    └── uie_fastunet_base32_benchEUVP/
        ├── best.pt
        └── last.pt
