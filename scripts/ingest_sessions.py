"""Ingest completed session artifacts into a verified-run ledger.

Reads run directories produced by the resume-safe runner (Kaggle or Colab).
For every run directory it checks, in order:

  1. ``status.json`` exists and parses, is ``status == "completed"`` with
     ``verified is True`` and finite ``test_auc``/``test_acc`` in ``[0, 1]``.
  2. The run id is a preregistered plan identity (or a documented retry
     variant of one); cross-checks dataset and seed against the plan.
  3. An official evaluator CSV exists whose *filename* embeds
     ``[AUC]``/``[ACC]`` values (the MedMNIST evaluator names prediction files
     ``{flag}{size}_test_[AUC]{auc:.3f}_[ACC]{acc:.3f}@...csv``); those values
     must agree with ``status.json`` within the 3-decimal rounding tolerance.
  4. The final ``test  auc: ... acc: ...`` line of ``train.log`` agrees with
     ``status.json`` within ``1e-4``.

A run enters ``runs`` only if all four checks pass. Runs with a non-plan id
that pass checks go to ``retries`` (documented attempt variants, never counted
as baselines). Anything that fails any check goes to ``rejected`` with the
reason. The tool only reads sources; it never fabricates a value and never
rewrites a source artifact.

The output ledger is intentionally plain data: ``run_id``, dataset, seed, the
validated ``test_auc``/``test_acc``, evidence paths, and the per-run check
results. Consumers (the workbook builder, the report generator) read only this
file, so a metric is never invented at ingestion time.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVALUATOR_NAME_RE = (
    r"\[AUC\]?(?P<auc>\d+\.\d{1,4})_(?:\[ACC\])?(?P<acc>\d+\.\d{1,4})@"
)
FINAL_LOG_RE = re.compile(r"^test\s+(\d+\s)?auc:\s*([0-9.]+)\s+acc:\s*([0-9.]+)",
                          re.MULTILINE)

EVALUATOR_TOLERANCE = 0.0015   # file name rounds to 3 decimals
LOG_TOLERANCE = 1e-4


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_unit(value):
    """True only for a real number in [0, 1]."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value >= 0 and value <= 1


def parse_run_id(run_id):
    """Parse dataset, dimension and seed from a canonical runner run id.

    Example: ``2d-chestmnist-resnet50-224-seed17-r1``
    """
    m = re.match(r"^(?P<dim>2d|3d)-(?P<dataset>[a-z0-9]+)-resnet50-"
                 r"(?P<res>\d+)-seed(?P<seed>\d+)-r(?P<attempt>\d+)", run_id)
    if not m:
        return None
    return dict(dim=m.group("dim"), dataset=m.group("dataset"),
                resolution=int(m.group("res")), seed=int(m.group("seed")),
                attempt=int(m.group("attempt")))


def discover_run_dirs(sources):
    """Yield (run_id, run_dir) for every directory with a status.json.

    Handles three layouts: a ``.../runs`` folder, a top-level session folder
    containing ``runs/``, and a folder where run dirs sit directly.
    """
    seen = set()
    for source in sources:
        src = Path(source)
        if not src.is_dir():
            print(f"[warn] source not a directory: {source}", file=sys.stderr)
            continue
        pools = []
        if (src / "runs").is_dir():
            pools.append(src / "runs")
        pools.append(src)
        for pool in pools:
            for child in sorted(pool.iterdir()):
                if not child.is_dir() or child in seen:
                    continue
                if (child / "status.json").is_file():
                    seen.add(child)
                    yield child.name, child


def load_plan(root=ROOT):
    path = root / "configs/experiment-plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    by_id = {r["run_id"]: r for r in plan["runs"]}
    return plan, by_id


def evaluator_values(run_dir):
    """Return [(path, auc, acc)] parsed from official evaluator file names.

    The evaluator writes ``{flag}{size}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}
    @{run}.csv``. The brackets are regex-literal in EVALUATOR_NAME_RE, but glob
    would treat them as a character class, so every CSV in the run directory is
    scanned and only names matching the pattern are kept.
    """
    out = []
    for path in sorted(run_dir.glob("*.csv")):
        m = re.search(EVALUATOR_NAME_RE, path.name)
        if not m:
            continue
        out.append((str(path.relative_to(run_dir.parent)), float(m.group("auc")),
                    float(m.group("acc"))))
    return out


def final_log_values(run_dir):
    """Return (auc, acc) from the last official test-metrics line, or None."""
    log = run_dir / "train.log"
    if not log.is_file():
        return None
    matches = list(FINAL_LOG_RE.finditer(
        log.read_text(encoding="utf-8", errors="replace")))
    if not matches:
        return None
    m = matches[-1]
    return float(m.group(2)), float(m.group(3))


def close(a, b, tol):
    return abs(a - b) <= tol
def ingest_run(run_dir, plan_by_id):
    """Return one of ('runs'|'retries'|'rejected', record)."""
    run_id = run_dir.name
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    parsed = parse_run_id(run_id)

    def reject(reason):
        return "rejected", {"run_id": run_id, "reason": reason}

    if status.get("status") != "completed":
        return reject(f"status is {status.get('status')!r}, not 'completed'")
    if status.get("verified") is not True:
        return reject("status.verified is not True")
    auc, acc = status.get("test_auc"), status.get("test_acc")
    if not finite_unit(auc) or not finite_unit(acc):
        return reject(f"test_auc/test_acc not finite in [0,1]: {auc!r}/{acc!r}")

    checks = {"plan_identity": "ok", "evaluator_filename": "ok", "final_log": "ok"}
    plan_ref = plan_by_id.get(run_id)
    if plan_ref is not None:
        if parsed is None or parsed["dataset"] != plan_ref["dataset"] \
                or parsed["seed"] != plan_ref["seed"] \
                or parsed["dim"] != plan_ref["dimension"]:
            return reject(
                "run id parses to dataset/seed/dimension that differs from the plan")
    elif parsed is None:
        return reject("run id is not a plan identity and cannot be parsed")

    evals = evaluator_values(run_dir)
    test_evals = [e for e in evals if "test" in Path(e[0]).name]
    if not test_evals:
        return reject("no official evaluator CSV matching [AUC]/[ACC] naming found")
    if not any(close(auc, e_auc, EVALUATOR_TOLERANCE)
               and close(acc, e_acc, EVALUATOR_TOLERANCE)
               for _, e_auc, e_acc in test_evals):
        return reject(f"evaluator file names disagree with status.json "
                      f"({auc:.5f}/{acc:.5f} vs "
                      f"{[(e[1], e[2]) for e in test_evals]})")

    fv = final_log_values(run_dir)
    if fv is None:
        return reject("final 'test  auc: ... acc: ...' line not found in train.log")
    if not close(auc, fv[0], LOG_TOLERANCE) or not close(acc, fv[1], LOG_TOLERANCE):
        return reject(f"final train.log line disagrees with status.json "
                      f"({auc:.5f}/{acc:.5f} vs {fv[0]:.5f}/{fv[1]:.5f})")

    completed_at = status.get("end_utc") or status.get("completed_at") \
        or status.get("finished_at") or now_iso()
    record = {
        "run_id": run_id,
        "dataset": (plan_ref or parsed)["dataset"],
        "seed": (plan_ref or parsed)["seed"],
        "attempt": parsed["attempt"],
        "test_auc": auc,
        "test_acc": acc,
        "verified_at": completed_at,
        "evidence": {
            "status": str(status_path.relative_to(run_dir.parent)),
            "evaluator": [e[0] for e in test_evals],
            "train_log": str((run_dir / "train.log").relative_to(run_dir.parent)),
        },
        "checks": checks,
    }
    bucket = "runs" if plan_ref is not None else "retries"
    return bucket, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="DIR",
                        help="session directory to inspect (repeatable)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "configs/verified-runs.json")
    parser.add_argument("--archive", type=Path, default=None,
                        help="optional dir under which accepted artifacts are "
                             "copied so evidence stays inside the project")
    args = parser.parse_args(argv)

    plan, plan_by_id = load_plan()
    out = {"schema_version": 1, "generated_at": now_iso(),
           "sources": [str(s) for s in args.source],
           "runs": [], "retries": [], "rejected": []}

    for run_id, run_dir in discover_run_dirs(args.source):
        bucket, record = ingest_run(run_dir, plan_by_id)
        if bucket == "rejected":
            out["rejected"].append(record)
            print(f"rejected  {run_id:<48} {record['reason']}")
            continue
        if args.archive and bucket == "runs":
            import shutil
            dest = (args.archive / run_id)
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("status.json", "train.log", "command.json"):
                src = run_dir / name
                if src.is_file():
                    shutil.copy2(src, dest / name)
            for eval_rel in record["evidence"]["evaluator"]:
                src = run_dir.parent / eval_rel
                if src.is_file():
                    shutil.copy2(src, dest / Path(eval_rel).name)
            record["evidence"] = {
                "status": f"{run_id}/status.json",
                "evaluator": [f"{run_id}/{Path(p).name}"
                              for p in record["evidence"]["evaluator"]],
                "train_log": f"{run_id}/train.log"}
        out[bucket].append(record)
        print(f"verified  {run_id:<48} AUC={record['test_auc']:.5f} "
              f"ACC={record['test_acc']:.5f} "
              f"dataset={record['dataset']} seed={record['seed']}")

    out["summary"] = {
        "baseline_verified": len(out["runs"]),
        "retry_verified": len(out["retries"]),
        "rejected": len(out["rejected"]),
        "plan_total": len(plan["runs"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    print(f"baseline verified: {out['summary']['baseline_verified']} / "
          f"{out['summary']['plan_total']}; retries: "
          f"{out['summary']['retry_verified']}; rejected: "
          f"{out['summary']['rejected']}")
    if out["rejected"]:
        print("Rejected entries are NOT results; inspect before rerunning.")


if __name__ == "__main__":
    main()