"""Aggregate MedMNIST native-28 run results into the meta-study summary.

Reads runs/official28/results.jsonl (one record per run, written by
run_official_2d_28.py) and emits:
  - runs/official28/summary.json : per-dataset n, mean/std of test AUC & ACC
    across seeds, min/max spread, and a stability flag.
  - runs/official28/summary.md   : human-readable table.

Stability metrics (for the leaderboard-instability paper):
  - spread = max - min across seeds (per dataset, per metric)
  - cv = std / mean
A dataset is flagged "unstable" when the seed spread in test AUC exceeds
0.01 (1pp) -- a gap larger than many published single-seed "wins".
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "official28" / "results.jsonl"
UNSTABLE_SPREAD_AUC = 0.01


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"no results yet: {RESULTS} missing")
    runs = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in runs if r.get("exit_code") == 0 and r.get("test_auc") is not None]
    print(f"runs: {len(runs)} total, {len(ok)} completed with metrics")

    by_ds = defaultdict(list)
    for r in ok:
        by_ds[r["dataset"]].append(r)

    summary = {}
    for ds, rs in sorted(by_ds.items()):
        aucs = [r["test_auc"] for r in rs]
        accs = [r["test_acc"] for r in rs]
        spread = max(aucs) - min(aucs)
        summary[ds] = {
            "n_seeds": len(rs),
            "seeds": sorted(r["seed"] for r in rs),
            "test_auc_mean": round(mean(aucs), 5),
            "test_auc_std": round(std(aucs), 5),
            "test_auc_min": round(min(aucs), 5),
            "test_auc_max": round(max(aucs), 5),
            "test_auc_spread": round(spread, 5),
            "test_acc_mean": round(mean(accs), 5),
            "test_acc_std": round(std(accs), 5),
            "unstable_seed_spread": spread > UNSTABLE_SPREAD_AUC,
        }

    (ROOT / "runs" / "official28" / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    lines = ["# MedMNIST native-28 run summary", "",
             f"Completed runs with metrics: {len(ok)} / {len(runs)}", "",
             "| dataset | n | test AUC mean±std | spread | unstable? |",
             "|---|---|---|---|---|"]
    for ds, s in summary.items():
        flag = "YES" if s["unstable_seed_spread"] else "no"
        lines.append(f"| {ds} | {s['n_seeds']} | {s['test_auc_mean']}±{s['test_auc_std']} | "
                     f"{s['test_auc_spread']} | {flag} |")
    (ROOT / "runs" / "official28" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
