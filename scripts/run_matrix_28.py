"""Generate the native-28x28 run matrix for MedMNIST v2 2D reproducibility runs.

Reads configs/experiment-plan.json, emits configs/run-matrix-28.json:
60 runs = 12 MedMNIST2D datasets x resnet18 x 5 seeds, native 28x28 input
(no --resize), 100 epochs, batch 128, Adam lr=0.001, MultiStepLR(50,75,gamma=0.1).

Rationale (see docs/PROTOCOL_DECISION_28.md): the earlier 224x224 ResNet-50
protocol measured ~1,330 s/epoch on ChestMNIST (~37 GPU-h per 100-epoch run),
which puts the full 2D suite at ~1,200 T4 GPU-hours -- infeasible on the
30 h/week free quota. Native 28x28 matches the MedMNIST paper's standard
28x28 benchmark column and makes the full multi-seed study feasible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [17, 29, 43, 71, 101]
ARCH = "resnet18"
RESOLUTION = 28
EPOCHS = 100
BATCH_SIZE = 128
LR = 0.001
SCHEDULE = {"type": "MultiStepLR", "milestones": [50, 75], "gamma": 0.1}
UPSTREAM_COMMIT = "70b6b3a7ad7afddff1df2a3b735235830fbdb142"


def config_hash(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def main() -> None:
    plan = json.loads((ROOT / "configs" / "experiment-plan.json").read_text(encoding="utf-8"))
    datasets = [d for d in plan["datasets"] if d["dimension"] == "2d"]
    assert len(datasets) == 12, f"expected 12 2D datasets, got {len(datasets)}"

    runs = []
    for ds in datasets:
        for seed in SEEDS:
            core = {
                "dataset": ds["flag"],
                "arch": ARCH,
                "resolution": RESOLUTION,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "schedule": SCHEDULE,
                "optimizer": "Adam",
                "seed": seed,
                "upstream_commit": UPSTREAM_COMMIT,
                "resize": False,
                "as_rgb": True,
            }
            run_id = f"2d-{ds['flag']}-{ARCH}-28-seed{seed}"
            runs.append({"run_id": run_id, "config_hash": config_hash(core), **core})

    matrix = {
        "protocol": "medmnist2d-native28-resnet18",
        "n_runs": len(runs),
        "runs": runs,
        "note": "Native 28x28 (no --resize). See docs/PROTOCOL_DECISION_28.md.",
    }
    out = ROOT / "configs" / "run-matrix-28.json"
    out.write_text(json.dumps(matrix, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(runs)} runs")


if __name__ == "__main__":
    main()
