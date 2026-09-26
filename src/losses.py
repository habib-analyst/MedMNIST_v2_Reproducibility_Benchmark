"""Task-appropriate losses and online loss-scale normalization."""
import math

import torch
from torch.nn import functional as F


def task_loss(logits, targets, task):
    if task == 'multi-label, binary-class':
        target = targets.to(dtype=logits.dtype)
        if target.shape != logits.shape:
            raise ValueError(f'Multilabel target shape {tuple(target.shape)} must equal logits {tuple(logits.shape)}')
        return F.binary_cross_entropy_with_logits(logits, target)
    target = targets.reshape(-1).long()
    if target.shape[0] != logits.shape[0]:
        raise ValueError('Class target batch size does not match logits')
    return F.cross_entropy(logits, target)


class TaskLossNormalizer:
    """Normalize task losses with detached exponential moving averages."""
    def __init__(self, momentum=0.95, epsilon=1e-8):
        if not 0 <= momentum < 1:
            raise ValueError('momentum must be in [0, 1)')
        self.momentum = momentum
        self.epsilon = epsilon
        self.scales = {}

    def normalize(self, dataset, loss):
        measured = float(loss.detach())
        if not math.isfinite(measured) or measured < 0:
            raise ValueError('Loss scale requires a finite nonnegative loss')
        previous = self.scales.get(dataset, 1.0)
        current = self.momentum * previous + (1 - self.momentum) * measured
        self.scales[dataset] = current
        return loss / max(current, self.epsilon)
