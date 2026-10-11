[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/MedMNIST_v2_Reproducibility_Benchmark.svg)](https://github.com/habib-analyst/MedMNIST_v2_Reproducibility_Benchmark)

**Medical-imaging leaderboards are decided by single seeds. We re-ran MedMNIST v2 with 5 fixed seeds per dataset under a pinned, auditable protocol — 4/4 datasets tested so far are seed-unstable, with spreads up to ~3pp.**

## Quickstart

```bash
git clone https://github.com/habib-analyst/MedMNIST_v2_Reproducibility_Benchmark.git
cd MedMNIST_v2_Reproducibility_Benchmark
pip install -r requirements.txt
```

Rebuild the released batch-1 aggregates from the recorded run results:

```bash
python scripts/aggregate_results.py
```

Expected output: writes `runs/official28/summary.json` and `runs/official28/summary.md` — per-dataset 5-seed mean±std of test AUC/ACC, min/max seed spread, and a stability flag per dataset (flagged unstable when seed spread in test AUC exceeds 1pp).

Full protocol (what produced the results): seeds 17/29/43/71/101, ResNet-18 on native 28x28 inputs, 100 epochs, Adam lr=1e-3, MultiStepLR at epochs 50/75 (gamma 0.1), batch size 128, pinned upstream MedMNIST experiments commit `70b6b3a7ad7afddff1df2a3b735235830fbdb142`. Batch runners live in `kaggle/` and `scripts/run_official_2d_28.py`.

## Measured results

Batch 1 (native-28 ResNet-18, 100 epochs, 5 seeds per dataset) — test AUC:

| Dataset | Mean ± std | Seed spread | Verdict |
|---|---|---|---|
| breast | 0.9074 ± 0.0103 | 0.0295 | UNSTABLE |
| pneumonia | 0.9464 ± 0.0106 | 0.0272 | UNSTABLE |
| retina | 0.7290 ± 0.0081 | 0.0255 | UNSTABLE |
| derma (batch 2b) | 0.91305 ± 0.00484 | 0.0091 | below flag threshold |
| derma (batch 2c, 5/5 runs) | 0.91535 ± 0.00491 | 0.0135 (1.35pp) | UNSTABLE |

**Verdict: 4/4 datasets unstable-flagged.** Seed spreads of ~3pp exceed many published single-seed "wins" — a ranking flip, not noise. This finding motivated [stable-ssl](https://github.com/habib-analyst/stable-ssl), which treats seed instability as a training problem.

Status honesty: batch 1 is complete (15/15 runs, `runs/official28/`). The remaining 2D datasets (45 runs) and 3D suite (30 runs, 5 verified complete: AdrenalMNIST3D x5) are not done — partial runs are kept as separate labeled records, never merged into claims. The 224x224 ResNet-50 arm proved infeasible on free-tier compute (27- and 32-epoch records retained and labeled). 3D timing measured: ~120.9–131.9 s/epoch on Kaggle Tesla T4 x2.

## How it works

1. **Pin the source** — all official training uses a fixed upstream commit.
2. **Fix the protocol** — seeds, epochs, optimizer, schedule, batch size specified once (`run-plan.json`), applied uniformly.
3. **Keep evidence** — logs, exact commands, status records, environment freezes under `kaggle/evidence/`; partial runs preserved, never silently completed.
4. **Report per-dataset** — aggregates are 5-seed mean±std plus min/max spread per dataset; no single average that hides failure modes.
5. **Separate exploration** — partial-checkpoint evaluations live in `exploratory/` and are excluded from official aggregates.

Phase 2 (unified learning across modalities) is deliberately deferred until Phase 1 is complete; its code is a reference design with no real-data results.

## Citation

```bibtex
@Article{medmnistv2,
    author  = {Jiancheng Yang and Rui Shi and Donglai Wei and Zequan Liu and
               Lin Zhao and Bilian Ke and Hanspeter Pfister and Bingbing Ni},
    title   = {MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
               biomedical image classification},
    journal = {Scientific Data},
    year    = {2023},
    volume  = {10},
    number  = {41},
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
