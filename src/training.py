"""One-step optimization and shared-gradient monitoring."""
import math

import torch
from torch.nn import functional as F

from .losses import task_loss


def train_step(model, optimizer, dataset, inputs, targets, spec, normalizer=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs, dataset)
    raw_loss = task_loss(logits, targets, spec.task)
    loss = normalizer.normalize(dataset, raw_loss) if normalizer else raw_loss
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    optimizer.step()
    value = float(raw_loss.detach())
    return {'dataset': dataset, 'loss': value, 'loss_finite': math.isfinite(value), 'batch_size': inputs.shape[0]}


def task_gradient_cosines(losses, shared_parameters):
    parameters = [parameter for parameter in shared_parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError('At least one trainable shared parameter is required')
    vectors = {}
    items = list(losses.items())
    for index, (name, loss) in enumerate(items):
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=index < len(items) - 1, allow_unused=True,
        )
        vectors[name] = torch.cat([
            (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ])
    return {
        left: {
            right: float(F.cosine_similarity(vectors[left], vectors[right], dim=0).detach())
            for right in vectors
        }
        for left in vectors
    }
