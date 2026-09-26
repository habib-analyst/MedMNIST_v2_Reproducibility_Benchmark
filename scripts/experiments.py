"""Evidence-free run planning and strict aggregation of verified records."""
import argparse
import ast
import json
import math
import statistics
import subprocess
from pathlib import Path

SEEDS = (17, 29, 43, 71, 101)
ROOT = Path(__file__).resolve().parents[1]


def patch_3d_checkpoint(source):
    """Return the minimal upstream compatibility patch; refuse unknown input."""
    old = "'net': model.state_dict(),"
    new = "'net': best_model.state_dict(),"
    if source.count(old) != 1 or new in source:
        raise ValueError('Expected exactly one unpatched 3D checkpoint write')
    return source.replace(old, new)


def build_plan(root=ROOT):
    tree = ast.parse((root/'external/medmnist/medmnist/info.py').read_text(encoding='utf-8'))
    info = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'INFO' for t in n.targets))
    references = json.loads((root/'configs/benchmark-references.json').read_text())
    datasets, runs = [], []
    for flag in sorted(info, key=lambda x: (x != 'chestmnist', x.endswith('3d'), x)):
        item = info[flag]
        dimension = '3d' if flag.endswith('3d') else '2d'
        resolution = 28 if dimension == '3d' else 224
        datasets.append(dict(flag=flag, name=item['python_class'], dimension=dimension,
            task=item['task'], classes=len(item['label']), channels=item['n_channels'],
            train_count=item['n_samples']['train'], val_count=item['n_samples']['val'],
            test_count=item['n_samples']['test'], resolution=resolution,
            reported_auc=references['values'][flag][0], reported_acc=references['values'][flag][1],
            source=references['source'], data_url=item['url'], md5=item['MD5'],
            license=item['license']))
        for seed in SEEDS:
            runs.append(dict(run_id=f'{dimension}-{flag}-resnet50-{resolution}-seed{seed}-r1',
                dataset=flag, seed=seed, attempt=1, dimension=dimension, resolution=resolution,
                source_size=28, epochs=100, batch_size=128 if dimension=='2d' else 32,
                learning_rate=.001, status='planned', auc=None, acc=None))
    commit = subprocess.check_output(['git','-C',str(root/'external/medmnist-experiments'),
                                     'rev-parse','HEAD'], text=True).strip()
    metadata_commit = subprocess.check_output(['git','-C',str(root/'external/medmnist'),
                                              'rev-parse','HEAD'], text=True).strip()
    return dict(seeds=list(SEEDS), upstream_commit=commit, metadata_commit=metadata_commit,
                datasets=datasets, runs=runs,
                note='Plan only. No measured outcomes.')


def summarize(rows):
    if len(rows) != 5 or {r['seed'] for r in rows} != set(SEEDS):
        raise ValueError('Exactly five distinct preregistered seeds are required')
    if any(r['status'] != 'verified' for r in rows):
        raise ValueError('Only verified runs may be summarized')
    datasets = {r.get('dataset') for r in rows}
    if len(datasets) != 1 or None in datasets:
        raise ValueError('All five runs must belong to one identified dataset')
    result = {}
    for metric in ('auc','acc'):
        values = [r[metric] for r in rows]
        if any(isinstance(v, bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError('Metrics must be finite numbers between zero and one')
        result[f'{metric}_mean'] = statistics.mean(values)
        result[f'{metric}_sd'] = statistics.stdev(values)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'configs/experiment-plan.json')
    args = parser.parse_args()
    plan = build_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    print(f'Planned {len(plan["runs"])} runs across {len(plan["datasets"])} datasets; no results created.')



