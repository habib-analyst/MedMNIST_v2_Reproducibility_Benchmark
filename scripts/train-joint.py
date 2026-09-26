"""Train one of the three joint MedMNIST settings; evaluation is separate."""
import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import CyclingLoaders, build_train_loaders
from src.losses import TaskLossNormalizer
from src.models import DimensionAwareModel, select_specs, specs_from_plan
from src.scheduling import temperature_probabilities
from src.training import train_step


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--mode', required=True, choices=['2d', '3d', 'unified'])
    value.add_argument('--datasets', nargs='+', help='Optional verified subset for pipeline smoke checks')
    value.add_argument('--epochs', type=int, default=100)
    value.add_argument('--steps-per-epoch', type=int)
    value.add_argument('--batch-size-2d', type=int, default=128)
    value.add_argument('--batch-size-3d', type=int, default=32)
    value.add_argument('--embed-dim', type=int, default=128)
    value.add_argument('--depth', type=int, default=2)
    value.add_argument('--heads', type=int, default=4)
    value.add_argument('--temperature', type=float, default=0.0)
    value.add_argument('--learning-rate', type=float, default=1e-3)
    value.add_argument('--seed', type=int, default=17)
    value.add_argument('--download', action='store_true')
    value.add_argument('--output-root', type=Path, default=ROOT / 'runs' / 'joint')
    return value


def main():
    args = parser().parse_args()
    if args.epochs < 1 or min(args.batch_size_2d, args.batch_size_3d, args.embed_dim, args.depth, args.heads) < 1:
        raise SystemExit('Epochs, batch sizes and model dimensions must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    plan_path = ROOT / 'configs' / 'experiment-plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    all_specs = specs_from_plan(plan)
    specs = select_specs(all_specs, args.mode, args.datasets)
    loaders, counts = build_train_loaders(
        specs, ROOT / 'data', args.batch_size_2d, args.batch_size_3d, args.download,
    )
    streams = CyclingLoaders(loaders)
    probabilities = temperature_probabilities(counts, args.temperature)
    names = list(probabilities)
    weights = [probabilities[name] for name in names]
    steps_per_epoch = args.steps_per_epoch or sum(
        (count + (args.batch_size_2d if specs[name].dimension == '2d' else args.batch_size_3d) - 1)
        // (args.batch_size_2d if specs[name].dimension == '2d' else args.batch_size_3d)
        for name, count in counts.items()
    )
    model = DimensionAwareModel(specs, args.mode, args.embed_dim, args.depth, args.heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    normalizer = TaskLossNormalizer()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_root / f'{args.mode}-{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    status = {
        'kind': 'joint-training', 'status': 'running', 'mode': args.mode,
        'seed': args.seed, 'epochs_planned': args.epochs, 'epochs_completed': 0,
        'steps_per_epoch': steps_per_epoch, 'dataset_probabilities': probabilities,
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'official_baseline_runs_completed': 0,
        'metrics': None,
    }
    status_path = output / 'status.json'
    status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    generator = random.Random(args.seed)
    try:
        for epoch in range(args.epochs):
            losses = []
            for _ in range(steps_per_epoch):
                name = generator.choices(names, weights=weights, k=1)[0]
                inputs, targets = streams.next(name)
                record = train_step(model, optimizer, name, inputs, targets, specs[name], normalizer)
                losses.append(record['loss'])
            checkpoint = {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'mode': args.mode, 'specs': {name: vars(spec) for name, spec in specs.items()},
                'epoch': epoch + 1, 'seed': args.seed,
            }
            torch.save(checkpoint, output / 'last-checkpoint.pt')
            status.update(
                epochs_completed=epoch + 1,
                last_epoch_training_loss_mean=sum(losses) / len(losses),
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        status.update(status='completed', ended_utc=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        raise
    print(output)


if __name__ == '__main__':
    main()
