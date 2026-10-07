# Protocol decision: native 28x28 ResNet-18 for the feasible multi-seed study

Date: 2026-10-07. Author: Worker B (research agenda Project #6).

## The feasibility problem

The repo's original Phase 1 protocol (ResNet-50, 28x28 resized to 224x224 RGB,
100 epochs, Adam 1e-3, MultiStepLR@50/75) measured **~1,330 s/epoch** on
ChestMNIST on a Kaggle T4, i.e. **~37 GPU-hours per single 100-epoch run**
(`docs/RUNTIME_AND_VRAM.md`, README "Current status and evidence").

Extrapolated full cost of the 90-run plan at 224x224: **~1,150-1,200 T4
GPU-hours for the 2D suite alone** (README). At the binding free-quota
constraint (Kaggle ~30 GPU-h/week), that is ~40 weeks for 2D -- before any 3D
runs. The plan as specified is not executable.

## Decision

Run the multi-seed study at **native 28x28 with ResNet-18** (upstream
`train_and_eval_pytorch.py` at the same pinned commit
`70b6b3a7ad7afddff1df2a3b735235830fbdb142`, `--size 28`, no `--resize`,
`--model_flag resnet18`, identical optimizer/schedule/epochs/seeds).

Why this is scientifically sound, not a corner cut:

1. Native 28x28 is the MedMNIST paper's standard benchmark column (the
   configuration the published leaderboard everyone cites was produced in).
   Comparing our multi-seed statistics against the published 28x28 reference
   values is the *most* apples-to-apples literature comparison available.
2. The instability claim ("rankings unstable under seed variation") does not
   depend on input resolution; it depends on having ≥5 seeds per dataset.
3. Upstream's default `--model_flag` is `resnet18` and `--size` defaults to 28;
   we are using the authors' own default configuration, not a custom one.

## Honest bookkeeping

- The two partial 224x224 ChestMNIST records (27-epoch, 32-epoch) remain in
  `kaggle/evidence/` labeled as partial/exploratory. They are NOT merged into
  the native-28 aggregates.
- `configs/run-matrix-28.json` is the binding run list for the feasibility
  study (60 runs: 12 datasets x 5 seeds). The original `run-plan.json`
  (224x224) is retained as the historical record of the infeasible plan.
- If/when larger compute becomes available, the 224x224 ResNet-50 arm can be
  re-run; the matrix generator is parameterized for that.

## Cost estimate (native 28x28, ResNet-18, T4)

| scope | runs | est. GPU-h |
|---|---|---|
| 2D suite (12 datasets x 5 seeds) | 60 | ~45-60 |
| 3D suite (6 datasets x 5 seeds, ResNet-18 3D, pending calibration) | 30 | ~45-60 (to verify) |
| total | 90 | ~90-120 over ~4 Kaggle weeks |

Batch 1 (verification): breastmnist + retinamnist + pneumoniamnist x 5 seeds
= 15 runs, ~4-6 GPU-h, one Kaggle session.

## Follow-on recommendation (for director/parent)

The core paper claim needs ≥2 architectures per dataset to measure *rank
turnover*. Recommended Phase 1b: repeat the 60-run matrix with ResNet-50 @28x28
(~45-60 more GPU-h). This doubles the 2D cost; batch-2 decision point.
