# Technical status

## Prepared

- 90-run protocol and dataset metadata.
- 2D/3D reference code and pinned upstream source commit.
- Deterministic seeding, checkpointing, and resume-safe runners.
- Literature review and follow-on unified-learning proposal.
- Separated execution evidence and result snapshots.
- Batch-1 2D evidence: 15/15 native-28 ResNet-18 runs with verified five-seed aggregates (`runs/official28/`).
- 3D suite executed: 5/30 runs verified complete (AdrenalMNIST3D x5 seeds); per-run records under `kaggle/evidence/3d_adrenal_batch1/`.

## Not yet complete

- Remaining 70 official runs (45 2D + 25 3D).
- 3D peak-VRAM measurement (runtime measured: ~126 s/epoch avg on T4 x2).
- Five-seed aggregates for the remaining 15 datasets.
- Final container image and checkpoint release.
- Unified-learning experimental results.

## Evidence policy

Partial records are retained for transparency. They are labeled by observed epoch count and are not merged or presented as final benchmark results.
