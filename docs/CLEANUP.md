# Repository cleanup

The cleanup retains training/inference/evaluation code, original split manifests, unique checkpoint states, configuration and dependency files, tests, and paper results. Unique best/last/stage/export checkpoints are retained because their equivalence to the paper-selected weights has not been established. Original manifests are retained even when filenames are aliases, because scripts and checkpoint metadata reference them. Local absolute paths still need remapping.

Removed files remain recoverable from Git history at commit `f8ed1e02745af74385a04aa6d4d733608b071ce8`; history has not been rewritten.

## Removed files

- `Resources/resources.md`
- `live_video.py`
- `outputs/Test1_model.png`
- `outputs/Test1_raw.png`
- `outputs/Test2_model.png`
- `outputs/Test2_raw.png`
- `outputs/Test3_model.png`
- `outputs/Test3_raw.png`
- `outputs/Test4_model.png`
- `outputs/Test4_raw.png`
- `outputs/Test5_model.png`
- `outputs/Test5_raw.png`
- `outputs/Test6_model.png`
- `outputs/Test6_raw.png`
- `outputs/Test7_model.png`
- `outputs/Test7_raw.png`
- `outputs/readme.md`
- `scripts/count_everything.sh`
- `scripts/enhance_video.py`
- `uie/edge_osa_models.py`

Removed categories: generated caches, empty/placeholder documentation, redundant visual assets, deployment/demo utilities outside the paper reproduction workflow, and unused alternative/copied model code. Training SLURM scripts and configuration examples are retained for provenance.

Also removed `test_custom_data.py`, a standalone duplicate model/demo with notebook-only `!pip` syntax. Use `python -m scripts.infer_folder` instead.
