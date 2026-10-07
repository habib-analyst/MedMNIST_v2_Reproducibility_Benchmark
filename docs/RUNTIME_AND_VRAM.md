# Runtime and VRAM status
| Record | Dataset | Type | Model | Hardware | VRAM | Batch | Epochs | Seconds/epoch | Evidence |
|---|---|---|---|---|---|---:|---:|---:|---|
| Historical 32-epoch record | ChestMNIST | 2D | ResNet-50, 224x224 | GPU model not fully recorded | Not measured | 128 | 100 planned | ~1,325 | `kaggle/evidence/gpu_historical_32epoch_attempt/train.log` |
| 27-epoch record | ChestMNIST | 2D | ResNet-50, 224x224 | Kaggle GPU; exact model not retained | Not measured | 128 | 100 planned | ~1,341 | `kaggle/evidence/gpu_partial_27epoch_record/status.json` |
| 3D calibration (seed 17) | AdrenalMNIST3D | 3D | ResNet-50-based 3D | Kaggle (Tesla T4 x2) | Not measured | Not recorded | 100 | ~120.9 | Measured 2026-10-07 |
| 3D calibration (seed 29) | AdrenalMNIST3D | 3D | ResNet-50-based 3D | Kaggle (Tesla T4 x2) | Not measured | Not recorded | 100 | ~131.9 | Measured 2026-10-07 |
The 2D values are observations from two separate records. The 100-epoch duration of 36.8–37.3 GPU-hours is an extrapolation. The 3D values are measured calibration runs recorded on 2026-10-07.
Hardware note: the Kaggle free-tier GPU pool is T4-class (16GB). The exact model instance was not recorded in the evidence metadata, so this repository does not claim a specific GPU model.
Derived suite estimate (not measured): scaling the observed ChestMNIST epoch time by each 2D dataset's training-set size gives approximately 1,150–1,200 T4-class GPU-hours for the full 2D suite (60 runs × 100 epochs). This is a planning estimate for allocation sizing, not a measurement.
## Measured 3D calibration (2026-10-07)
- Measured on Kaggle (Tesla T4 x2): AdrenalMNIST3D, ResNet-50 3D, 100 epochs — seed 17: 12,088.8 s total (~120.9 s/epoch); seed 29: 13,187.4 s total (~131.9 s/epoch).
- Average ~126 s/epoch ≈ 3.5 GPU-hours per complete 100-epoch 3D run.
- Full 3D suite (6 datasets × 5 seeds = 30 runs): ≈ 105 T4-class GPU-hours.
- Session note: 5 of 30 runs verified complete in one 11-hour Kaggle session (all adrenalmnist3d); 2 fracturemnist3d runs paused and resumable via checkpointing.
- Peak VRAM: not yet measured (per-epoch logs are written to worker-disk files not exposed in Kaggle's live log viewer).
- Full Phase 1 total (2D + 3D, estimate): ≈ 1,255–1,305 T4-hours ≈ 330–435 A100-hours (3–4× heuristic).
