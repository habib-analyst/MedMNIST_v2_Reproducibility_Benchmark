# Results status

| Scope | Planned | Completed | Status |
|---|---:|---:|---|
| Batch 1 (breast/retina/pneumonia, native-28 ResNet-18) | 15 | 15 | Five-seed mean±std per dataset; `runs/official28/` |
| Other 2D datasets (native-28 arm) | 45 | 0 | Not run |
| 224x224 ResNet-50 arm (historical) | 60 | 0 | 27-epoch + 32-epoch partial records retained |
| 3D datasets | 30 | 5 | AdrenalMNIST3D x5 verified; nodule seed 17 stalled 37/100; 2 fracture paused (Kaggle) |

Batch-1 five-seed test AUC (mean±std): breast 0.9074±0.0103, pneumonia 0.9464±0.0106, retina 0.7290±0.0081.

Published references are in `configs/benchmark-references.json`; they are not project measurements. Exploratory partial-checkpoint metrics are in `exploratory/` and are excluded from official aggregates.
