"""Synthetic shape and optimization smoke checks; never performance evidence."""
import random

import torch

from .losses import TaskLossNormalizer
from .models import DimensionAwareModel
from .training import train_step


def architecture_smoke(mode, specs, seed=17):
    random.seed(seed)
    torch.manual_seed(seed)
    model = DimensionAwareModel(specs, mode=mode, embed_dim=16, depth=1, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    normalizer = TaskLossNormalizer()
    steps = []
    for name in model.dataset_names:
        spec = model.specs[name]
        if spec.dimension == '2d':
            inputs = torch.randn(2, 3, 28, 28)
        else:
            inputs = torch.randn(2, 3, 12, 12, 12)
        if spec.task == 'multi-label, binary-class':
            targets = torch.randint(0, 2, (2, spec.classes))
        else:
            targets = torch.randint(0, spec.classes, (2,))
        steps.append(train_step(model, optimizer, name, inputs, targets, spec, normalizer))
    return {
        'kind': 'architecture-smoke-only',
        'status': 'completed',
        'mode': mode,
        'seed': seed,
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'datasets': list(model.dataset_names),
        'steps': steps,
        'official_runs_completed': 0,
        'caveat': 'Synthetic tensors verify routing, losses and optimization only. No AUC or ACC is computed.',
    }
