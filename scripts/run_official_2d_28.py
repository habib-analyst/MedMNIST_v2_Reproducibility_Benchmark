"""Launch one native-28x28 MedMNIST2D ResNet-18 run with deterministic seeding.

Uses the official upstream training script at the pinned commit
(via run_official_2d.ensure_env seed-injection machinery) but trains at
native 28x28 (no --resize) with model_flag=resnet18, per the feasible
protocol in docs/PROTOCOL_DECISION_28.md.

Writes: status.json, command.json, train.log, environment.txt, and appends
one record per run to results.jsonl with final test AUC/ACC parsed from the
training log plus seed, config hash, and environment info.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_official_2d import ensure_env, seed_everything  # noqa: E402

METRIC_RE = re.compile(r"test\s+auc:\s*([\d.]+)\s+acc:\s*([\d.]+)")
VAL_RE = re.compile(r"val\s+auc:\s*([\d.]+)\s+acc:\s*([\d.]+)")


def parse_final_metrics(log_text: str) -> dict:
    test_hits = METRIC_RE.findall(log_text)
    val_hits = VAL_RE.findall(log_text)
    out = {"test_auc": None, "test_acc": None, "val_auc": None, "val_acc": None}
    if test_hits:
        out["test_auc"], out["test_acc"] = float(test_hits[-1][0]), float(test_hits[-1][1])
    if val_hits:
        out["val_auc"], out["val_acc"] = float(val_hits[-1][0]), float(val_hits[-1][1])
    return out


def env_info() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)
    try:
        import medmnist
        info["medmnist"] = medmnist.__version__
    except Exception as e:  # pragma: no cover
        info["medmnist_error"] = str(e)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_id", required=True, help="run_id from configs/run-matrix-28.json")
    p.add_argument("--matrix", default=str(ROOT / "configs" / "run-matrix-28.json"))
    p.add_argument("--gpu", action="store_true", help="use gpu_ids=0 instead of CPU")
    p.add_argument("--download", action="store_true", default=True)
    p.add_argument("--results", default=None, help="results.jsonl path (default: runs/official28/results.jsonl)")
    args = p.parse_args()

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in matrix["runs"]}
    if args.run_id not in runs:
        raise SystemExit(f"run_id {args.run_id!r} not in matrix")
    cfg = runs[args.run_id]

    out = ROOT / "runs" / "official28" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results) if args.results else (ROOT / "runs" / "official28" / "results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    status = {
        "kind": "official-2d-native28",
        "run_id": args.run_id,
        "dataset": cfg["dataset"],
        "arch": cfg["arch"],
        "resolution": cfg["resolution"],
        "seed": cfg["seed"],
        "config_hash": cfg["config_hash"],
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    env = env_info()
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    seed_everything(cfg["seed"])
    env_dir = ensure_env(ROOT / "runs" / "_env_medmnist2d")
    gpu_ids = "0" if args.gpu else "-1"
    cmd = [
        sys.executable,
        str(env_dir / "train_and_eval_pytorch.py"),
        "--data_flag", cfg["dataset"],
        "--output_root", str(out / "output"),
        "--num_epochs", str(cfg["epochs"]),
        "--size", str(cfg["resolution"]),
        "--gpu_ids", gpu_ids,
        "--batch_size", str(cfg["batch_size"]),
        "--model_flag", cfg["arch"],
        # NOTE: no --resize -> native 28x28 training (feasible protocol)
        "--as_rgb",
        "--run", args.run_id,
        "--seed", str(cfg["seed"]),
    ]
    if args.download:
        cmd.append("--download")
    (out / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")

    log_path = out / "train.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("CMD " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(env_dir), stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed_s = time.time() - t0

    metrics = parse_final_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
    record = {
        "run_id": args.run_id,
        "config_hash": cfg["config_hash"],
        "dataset": cfg["dataset"],
        "arch": cfg["arch"],
        "resolution": cfg["resolution"],
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "optimizer": cfg["optimizer"],
        "schedule": cfg["schedule"],
        "upstream_commit": cfg["upstream_commit"],
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed_s, 1),
        "started_utc": status["started_utc"],
        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env,
        "log_path": str(log_path.relative_to(ROOT)),
        **metrics,
    }
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    status["status"] = "completed" if proc.returncode == 0 else "failed"
    status["exit_code"] = proc.returncode
    status["ended_utc"] = record["ended_utc"]
    status["metrics"] = metrics
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "metrics": metrics, "elapsed_s": record["elapsed_s"]}, indent=2))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
