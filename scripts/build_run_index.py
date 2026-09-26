"""Build the submission run index from the preregistered plan and recorded evidence.

The index contains identities and statuses only. It never fills a performance
value: a metric column exists solely for rows whose evidence is complete, and
no such row exists until a verified run artifact is ingested by the pipeline.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLUMNS = ['run_id', 'dataset', 'seed', 'dimension', 'resolution', 'batch_size',
           'kind', 'status', 'evidence_path', 'note']

VALID_STATUS = {'not-started', 'planned', 'running', 'completed', 'verified',
                'failed', 'interrupted'}


def attempt_rows(root=ROOT):
    """Local official-protocol attempts, read from filesystem evidence."""
    rows = []
    attempts_dir = root / 'runs' / 'attempts'
    if not attempts_dir.exists():
        return rows
    for directory in sorted(p for p in attempts_dir.iterdir() if p.is_dir()):
        status_path = directory / 'status.json'
        record = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        status = record.get('status', 'unknown')
        note = ''
        if (directory / 'RECOVERY.md').exists():
            status = 'interrupted'
            note = 'recorded status superseded by recovery note'
        rows.append({
            'run_id': directory.name,
            'dataset': record.get('dataset', 'chestmnist'),
            'seed': record.get('seed', 17),
            'dimension': '2d',
            'resolution': record.get('resolution', 224),
            'batch_size': record.get('batch_size', ''),
            'kind': 'official-attempt-local',
            'status': status,
            'evidence_path': f'runs/attempts/{directory.name}',
            'note': note or 'see train.log and status.json in the run directory',
        })
    return rows


def plan_rows(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    rows = []
    for run in plan['runs']:
        rows.append({
            'run_id': run['run_id'],
            'dataset': run['dataset'],
            'seed': run['seed'],
            'dimension': run['dimension'],
            'resolution': run['resolution'],
            'batch_size': run['batch_size'],
            'kind': 'official-baseline',
            'status': 'not-started',
            'evidence_path': '',
            'note': '',
        })
    return rows


def build_rows(root=ROOT):
    """Merge the plan with recorded attempts.

    A local attempt whose directory name equals a plan identity is folded into
    that identity's row, so every run id appears exactly once while the attempt
    evidence and its failure status remain visible. Other attempts (for example
    a second batch-size variant) keep their own row.
    """
    plan = plan_rows(root / 'configs/experiment-plan.json')
    attempts = attempt_rows(root)
    by_id = {row['run_id']: row for row in attempts}
    plan_ids = {row['run_id'] for row in plan}
    extra = [row for row in attempts if row['run_id'] not in plan_ids]
    for row in plan:
        attempt = by_id.get(row['run_id'])
        if attempt is not None:
            row['status'] = attempt['status']
            row['evidence_path'] = attempt['evidence_path']
            row['note'] = ('first local attempt failed; later attempts and the cloud run '
                           'with the same identity are indexed separately')
    rows = plan + extra
    ids = [row['run_id'] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f'duplicate run identities in index: {duplicates}')
    for row in rows:
        if row['status'] not in VALID_STATUS:
            raise ValueError(f'unknown status {row["status"]!r} for {row["run_id"]}')
    if any('auc' in key.lower() or 'acc' in key.lower() for key in COLUMNS):
        raise ValueError('the run index must not carry performance columns')
    return rows


def write_index(destination, root=ROOT):
    rows = build_rows(root)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'submission/artifacts_index/run_index.csv')
    args = parser.parse_args()
    path, rows = write_index(args.output)
    print(f'wrote {path} with {len(rows)} rows; no performance values present')
