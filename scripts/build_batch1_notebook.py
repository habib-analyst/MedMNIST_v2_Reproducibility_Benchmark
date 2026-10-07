"""Build the Kaggle batch-1 notebook for the native-28 MedMNIST reproducibility study.

Batch 1 (verification): breastmnist + retinamnist + pneumoniamnist x 5 seeds
= 15 runs, resnet18, native 28x28, 100 epochs. Estimated ~4-6 T4 GPU-hours.

Notebook settings required: GPU accelerator ON (T4), Internet ON (Zenodo
data download). Run all cells top to bottom in one session.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BATCH1_DATASETS = ["breastmnist", "retinamnist", "pneumoniamnist"]
BATCH1_SEEDS = [17, 29, 43, 71, 101]
UPSTREAM_COMMIT = "70b6b3a7ad7afddff1df2a3b735235830fbdb142"
BENCH_REPO = "https://github.com/habib-analyst/MedMNIST_v2_Reproducibility_Benchmark.git"

RUN_IDS = [f"2d-{d}-resnet18-28-seed{s}" for d in BATCH1_DATASETS for s in BATCH1_SEEDS]

CELL_INSTALL = """!pip install -q medmnist tensorboardX scikit-learn
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")"""

CELL_SETUP = f"""import os, subprocess
os.chdir("/kaggle/working")
if not os.path.exists("bench"):
    subprocess.run(["git", "clone", "--depth", "1", "{BENCH_REPO}", "bench"], check=True)
os.chdir("/kaggle/working/bench")
subprocess.run(["git", "pull", "--ff-only"], check=False)
ext = "external/medmnist-experiments"
if not os.path.exists(ext):
    os.makedirs("external", exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/MedMNIST/experiments.git", ext], check=True)
subprocess.run(["git", "-C", ext, "fetch", "--depth", "1", "origin", "{UPSTREAM_COMMIT}"], check=True)
subprocess.run(["git", "-C", ext, "checkout", "{UPSTREAM_COMMIT}"], check=True)
print("setup done; upstream at", subprocess.run(["git", "-C", ext, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())"""

CELL_RUN = """import json, subprocess, sys, time
run_ids = %s
results = "/kaggle/working/results_batch1.jsonl"
open(results, "a").close()
t0 = time.time()
for i, rid in enumerate(run_ids, 1):
    print(f"\\n===== [{i}/{len(run_ids)}] {rid} =====", flush=True)
    p = subprocess.run(
        [sys.executable, "scripts/run_official_2d_28.py", "--run_id", rid,
         "--gpu", "--results", results],
        cwd="/kaggle/working/bench", text=True)
    print(f"exit={p.returncode} elapsed_total={(time.time()-t0)/3600:.2f}h", flush=True)
    if p.returncode != 0:
        print("RUN FAILED -- continuing to next run")
print(f"\\nBATCH DONE in {(time.time()-t0)/3600:.2f}h. Results: {results}")""" % json.dumps(RUN_IDS)

CELL_SUMMARY = """import json
recs = [json.loads(l) for l in open("/kaggle/working/results_batch1.jsonl") if l.strip()]
print(f"{len(recs)} records")
for r in recs:
    print(r["run_id"], "auc=", r.get("test_auc"), "acc=", r.get("test_acc"), "exit=", r.get("exit_code"), f"{r.get('elapsed_s',0)/60:.1f}min")"""


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def main() -> None:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": [
            md_cell(
                "# MedMNIST native-28 reproducibility — Batch 1 (verification)\n\n"
                "**Protocol:** native 28x28, ResNet-18, 100 epochs, batch 128, Adam lr=0.001, "
                "MultiStepLR(50,75,γ=0.1), seeds 17/29/43/71/101. Upstream pinned at "
                f"`{UPSTREAM_COMMIT[:7]}`.\n\n"
                f"**Batch 1:** {len(RUN_IDS)} runs ({', '.join(BATCH1_DATASETS)} × 5 seeds), "
                "est. 4-6 T4 GPU-hours. Settings: **GPU ON, Internet ON**.\n\n"
                "After the run: download `/kaggle/working/results_batch1.jsonl` and the "
                "`runs/official28/` tree, and commit results to the benchmark repo."
            ),
            code_cell(CELL_INSTALL),
            code_cell(CELL_SETUP),
            code_cell(CELL_RUN),
            code_cell(CELL_SUMMARY),
        ],
    }
    out = ROOT / "kaggle" / "runner" / "medmnist_28_batch1_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(RUN_IDS)} runs)")


if __name__ == "__main__":
    main()
