# Runtime and VRAM status

| Record | Dataset | Type | Model | Hardware | VRAM | Batch | Epochs | Seconds/epoch | Evidence |
|---|---|---|---|---|---|---:|---:|---:|---|
| Historical 32-epoch record | ChestMNIST | 2D | ResNet-50, 224x224 | GPU model not fully recorded | Not measured | 128 | 100 planned | ~1,325 | `kaggle/evidence/gpu_historical_32epoch_attempt/train.log` |
| 27-epoch record | ChestMNIST | 2D | ResNet-50, 224x224 | Kaggle GPU; exact model not retained | Not measured | 128 | 100 planned | ~1,341 | `kaggle/evidence/gpu_partial_27epoch_record/status.json` |
| 3D calibration | Not yet run | 3D | ResNet-50-based 3D | TBD | TBD | TBD | TBD | TBD | Pending calibration |

The 2D values are observations from two separate records. The 100-epoch duration of 36.8–37.3 GPU-hours is an extrapolation. The 3D row must remain blank until a calibration run records it.

Hardware note: the Kaggle free-tier GPU pool is T4-class (16GB). The exact model instance was not recorded in the evidence metadata, so this repository does not claim a specific GPU model.

Derived suite estimate (not measured): scaling the observed ChestMNIST epoch time by each 2D dataset's training-set size gives approximately 1,150–1,200 T4-class GPU-hours for the full 2D suite (60 runs × 100 epochs). This is a planning estimate for allocation sizing, not a measurement.
