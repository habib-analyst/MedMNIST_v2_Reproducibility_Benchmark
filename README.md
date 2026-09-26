# MedMNIST v2 Reproducibility Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Early Stage](https://img.shields.io/badge/Status-Early%20Stage-orange.svg)](docs/TECHNICAL_STATUS.md)
[![Phase 1](https://img.shields.io/badge/Phase%201-In%20Progress-yellow.svg)](#phase-1-reproducibility-baseline)
[![Phase 2](https://img.shields.io/badge/Phase%202-Deferred-lightgrey.svg)](#phase-2-unified-learning-deferred)
[![Platform: Kaggle GPU](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF.svg)](kaggle/)
[![Upstream: MedMNIST](https://img.shields.io/badge/Upstream-MedMNIST%20v2-0A66C2.svg)](https://github.com/MedMNIST/MedMNIST)

A reproducible medical-image classification study on the public **MedMNIST v2** benchmark. This repository releases the protocol, reference implementation, configuration, literature review, execution evidence, and compute-aware plan for a two-phase research program.

---

## Author

**Habib Ur Rehman** -- BS Data Analytics (GCUF); machine learning and artificial intelligence.

| | |
|---|---|
| Website | [habib.top](https://www.habib.top) |
| GitHub | [github.com/habib-analyst](https://github.com/habib-analyst) |
| LinkedIn | [linkedin.com/in/hur-dev](https://www.linkedin.com/in/hur-dev) |

---

## Table of contents

1. [Project summary](#project-summary)
2. [Research aim](#research-aim)
3. [Why MedMNIST v2](#why-medmnist-v2)
4. [How this works](#how-this-works)
5. [Experimental protocol](#experimental-protocol)
6. [Current status and evidence](#current-status-and-evidence)
7. [Phase 1: Reproducibility baseline](#phase-1-reproducibility-baseline)
8. [Phase 2: Unified learning (deferred)](#phase-2-unified-learning-deferred)
9. [Impact](#impact)
10. [Reproduce from a fresh clone](#reproduce-from-a-fresh-clone)
11. [Compute-aware execution plan](#compute-aware-execution-plan)
12. [Repository layout](#repository-layout)
13. [Licensing and data](#licensing-and-data)

---

## Project summary

Medical-imaging models are frequently evaluated on a single dataset or a single random seed. Under those conditions it is difficult to separate a genuinely strong model from a lucky run or an implementation-specific advantage.

This project addresses that gap in two stages:

| Phase | Objective | Status |
|---|---|---|
| **Phase 1** | Reproduce 2D and 3D baseline performance across the full MedMNIST v2 suite under a fixed, auditable protocol (90 planned runs: 18 datasets x 5 seeds). | **Early-stage / in progress.** Not complete. |
| **Phase 2** | Investigate whether a shared, modality-aware model can generalize across heterogeneous 2D and 3D medical-imaging tasks. | **Deferred** until Phase 1 is complete. |

Execution to date was performed on **Kaggle GPU** notebooks using the official MedMNIST training scripts and a pinned upstream commit. Partial 2D evidence (27-epoch and 32-epoch records) is retained and labeled; no completed 100-epoch run and no five-seed aggregates are claimed.

---

## Research aim

The intended outcomes of the complete program are:

- reproducible baseline results under a fixed protocol;
- per-run variability, mean, and standard deviation across five seeds;
- comparison against published MedMNIST reference results;
- reusable checkpoints, configurations, logs, and evaluation code;
- an evidence-based assessment of unified learning across modalities and datasets.

Transparency of incomplete work is treated as a first-class requirement: partial records are preserved as separate, auditable artifacts and are never merged into final benchmark claims.

---

## Why MedMNIST v2

MedMNIST v2 provides standardized biomedical image datasets spanning multiple modalities, anatomical regions, resolutions, and classification tasks. That diversity makes it a suitable test bed for studying both reproducibility and the difficulty of transferring one representation across heterogeneous medical-image domains.

A model that performs well on one dataset should not automatically be assumed to generalize to another modality or anatomy. That observation motivates the later unified-learning phase and the insistence on per-dataset reporting rather than a single average score.

---

## How this works

### Protocol overview

1. **Pin the upstream source.** All official training uses MedMNIST experiments at commit `70b6b3a7ad7afddff1df2a3b735235830fbdb142`.
2. **Fix the protocol.** Seeds, epochs, optimizer, schedule, batch sizes, and input preprocessing are specified once and applied uniformly (see [Experimental protocol](#experimental-protocol)).
3. **Treat each run as an independent identity.** Outputs are per-run AUC and accuracy, then per-dataset aggregates across five seeds, then comparison to published references.
4. **Preserve evidence.** Training logs, exact commands, status records, environment freezes, and (where available) checksums are retained under `kaggle/evidence/`.
5. **Separate exploratory work.** Partial-checkpoint evaluations live in `exploratory/` and are excluded from official aggregates.
6. **Schedule against measured compute.** 2D timings inform planning estimates; 3D runtime and peak VRAM remain unmeasured pending a dedicated calibration run.

### Phase relationship

```text
Phase 1 (baseline)  -->  complete 90-run evidence + aggregates
                              |
                              v
Phase 2 (unified)   -->  compare shared model vs individual baselines
                         under identical splits, seeds, and evaluation
```

Phase 2 does not begin until Phase 1 deliverables are in place. The unified-learning code currently present is a reference design only; it has not produced real-data accuracy results.

---

## Experimental protocol

| Item | Configuration |
|---|---|
| Datasets | 12 MedMNIST2D + 6 MedMNIST3D |
| Runs | 5 seeds per dataset: 17, 29, 43, 71, 101 |
| Total planned runs | 90 |
| 2D model | ResNet-50; 28x28 inputs resized to 224x224 RGB |
| 3D model | ResNet-50-based 3D architecture |
| Batch size | 128 (2D), 32 (3D) |
| Epochs | 100 |
| Optimizer | Adam, learning rate 0.001 |
| Learning-rate schedule | MultiStepLR at epochs 50 and 75, gamma 0.1 |
| Pinned upstream source | MedMNIST experiments commit `70b6b3a7ad7afddff1df2a3b735235830fbdb142` |
| Execution platform (evidence) | Kaggle GPU (T4-class pool, 16 GB; exact instance not retained in metadata) |
| Data | Public; downloaded from the original MedMNIST distribution (not redistributed here) |

---

## Current status and evidence

**Status: early-stage. The full 90-run baseline is not complete and is not claimed to be complete.**

| Scope | Planned | Completed (100-epoch) | Notes |
|---|---:|---:|---|
| ChestMNIST seed 17 | 1 | 0 | 27/100 checkpointed execution retained |
| Other ChestMNIST seeds | 4 | 0 | Not run |
| Other 2D datasets | 55 | 0 | Not run |
| 3D datasets | 30 | 0 | Not run; notebook prepared, not executed |
| **Total Phase 1** | **90** | **0** | Baseline in progress |

### What has been observed

- A 2D ChestMNIST ResNet-50 execution reached **27 checkpointed epochs** and paused when the session budget could not start another epoch.
- A separate historical GPU execution record reached **32 epochs**.
- Both records include the training log, exact command, status record, and environment freeze.
- Measured 2D observation: approximately **1,325--1,341 seconds per epoch**.
- The **36.8--37.3 GPU-hour** figure for a complete 100-epoch ChestMNIST run is an **extrapolation**, not an end-to-end measurement.
- Derived planning estimate (not measured): scaling observed ChestMNIST epoch time by each 2D dataset's training-set size yields approximately **1,150--1,200 T4-class GPU-hours** for the full 2D suite (60 runs x 100 epochs).
- **3D runtime and peak VRAM are not yet measured.** A dedicated calibration run is required before the 3D schedule is finalized.

The full protocol remains pending because free GPU session and weekly compute limits were insufficient for the complete 2D/3D workload. Partial runs are retained as separate records rather than presented as final results.

### Evidence index

| Path | Contents |
|---|---|
| `kaggle/evidence/gpu_partial_27epoch_record/` | 27-epoch checkpointed execution: log, command, status, environment, progress |
| `kaggle/evidence/gpu_historical_32epoch_attempt/` | Separate 32-epoch GPU execution: log, command, status, environment, checksum |
| `kaggle/runner/` | GPU suite notebook and runner metadata (3D notebook prepared; not executed) |
| `kaggle/results/` | Small result snapshots from the executed workflow |
| `exploratory/` | Partial-checkpoint evaluation; explicitly outside official results |
| `docs/RESULTS_STATUS.md` | Tabular completion status |
| `docs/RUNTIME_AND_VRAM.md` | Timing and VRAM status (3D rows blank until calibrated) |
| `docs/TECHNICAL_STATUS.md` | Prepared vs. incomplete checklist |

No completed 100-epoch run is currently available; therefore no five-seed aggregate statistic is reported.

---

## Phase 1: Reproducibility baseline

Planned deliverables:

1. 90 trained checkpoints with a manifest and checksums.
2. Per-run and per-dataset AUC and accuracy.
3. Mean and standard deviation across the five seeds.
4. Comparison against published MedMNIST reference values (`configs/benchmark-references.json`).
5. A technical report documenting protocol, environment, and deviations.
6. A containerized environment for turnkey reproduction (planned; not yet tested -- see `docs/CONTAINER_PLAN.md`).

---

## Phase 2: Unified learning (deferred)

Phase 2 is deliberately deferred until the baseline is complete. The literature review supports the following design direction:

- modality-specific 2D/3D convolutional stems;
- a learned dataset or domain embedding;
- a shared Transformer-style encoder;
- dataset-specific prediction heads;
- task-appropriate losses;
- temperature-controlled dataset sampling to prevent large datasets from dominating;
- optional cross-dataset representation consistency.

The unified model will be compared against the individual baseline under identical splits, seeds, preprocessing, and evaluation. Results will be reported **per dataset**, not only as a single average, so that negative transfer is visible rather than hidden.

This phase is a research proposal at present. The code is a reference design and has not yet produced real-data accuracy results.

---

## Impact

Reliable medical-imaging research depends on baselines that other groups can audit, reproduce, and extend. A single published number -- without seeds, logs, environment freezes, or failure modes -- is not a sufficient foundation for method comparison or clinical-adjacent tooling.

This project is designed to contribute that foundation:

| Contribution | Who benefits | How |
|---|---|---|
| Fixed, auditable protocol | Benchmark authors and reviewers | Identical seeds, schedule, and upstream pin across all runs |
| Multi-seed statistics | Method developers | Mean / std across five seeds instead of a single lucky run |
| Released evidence artifacts | Reproducibility auditors | Logs, commands, status records, and environment freezes under version control |
| Honest incomplete-state reporting | Downstream users | Partial records labeled by epoch count; no silent inflation of completion |
| Per-dataset Phase 2 evaluation | Transfer-learning researchers | Negative transfer visible; no single average that hides failure modes |
| Permissive code license | Practitioners and educators | MIT-licensed protocol and runners for reuse, fine-tuning, and teaching |

Concrete research uses enabled by a completed Phase 1 release include:

- fair comparison against published MedMNIST reference results;
- reuse of reproducible checkpoints for fine-tuning or secondary evaluation;
- analysis of model stability across random seeds;
- transparent assessment of cross-modality and cross-dataset transfer once Phase 2 begins;
- compute planning for labs that face similar free-tier or shared-GPU constraints.

Until Phase 1 is complete, the primary contribution of this repository is the **protocol, tooling, and transparent evidence trail** -- not claimed final benchmark scores.

The original MedMNIST datasets remain subject to their respective licenses and are **not redistributed** in this repository.

---

## Reproduce from a fresh clone

```bash
git clone https://github.com/MedMNIST/experiments.git external/medmnist-experiments
git clone https://github.com/MedMNIST/MedMNIST.git external/medmnist
cd external/medmnist-experiments
git checkout 70b6b3a7ad7afddff1df2a3b735235830fbdb142
cd ../..
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Further detail:

- `docs/INSTALLATION.md` -- historical environment notes and current container status.
- `requirements-lock.txt` -- frozen historical Kaggle environment.
- `kaggle/runner/` -- GPU suite used for the 2D evidence; 3D notebook prepared but not executed.
- `scripts/run_official_2d.py` -- local CPU pilot runner.

---

## Compute-aware execution plan

1. Run one 3D calibration job and record GPU model, peak CUDA memory (`max_memory_allocated` / `max_memory_reserved`), batch size, input shape, and seconds per epoch.
2. Write measured values into `docs/RUNTIME_AND_VRAM.md`.
3. Use measured 2D and 3D timings to schedule datasets in parallel where storage and memory allow.
4. Preserve checkpoint/resume behavior and record every run in a machine-readable index.
5. Complete all five seeds per dataset.
6. Publish final artifacts separately from Git with manifests and checksums (`docs/ARTIFACT_POLICY.md`).

---

## Repository layout

| Path | Role |
|---|---|
| `kaggle/` | Exact GPU runner and Kaggle execution evidence |
| `src/` | Reference data, model, loss, training, and scheduling code |
| `configs/` | Experiment plan and benchmark references |
| `scripts/` | Local runners, safety checks, and orchestration |
| `exploratory/` | Partial-checkpoint evaluations (outside official results) |
| `literature/` | Literature review supporting the unified-learning phase |
| `docs/` | Status, timing, installation, and artifact policy |
| `licenses/` | Data and dependency licensing notes |
| `run-plan.json` | Machine-readable protocol summary |

---

## Licensing and data

| Asset | Terms |
|---|---|
| Project code and documentation in this repository | **MIT License** -- see [`LICENSE`](LICENSE) |
| MedMNIST datasets | **Not redistributed here**; remain under their original upstream terms |
| Large checkpoints, datasets, and event logs | Excluded from Git; to be released separately with manifests and checksums |

```text
MIT License
Copyright (c) 2026 Habib Ur Rehman
```

Full text: [`LICENSE`](LICENSE).

---

*Early-stage research software. Partial evidence is retained for auditability. No completed 100-epoch five-seed baseline is claimed at this time.*