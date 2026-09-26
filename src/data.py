"""MedMNIST input conversion and restartable per-dataset loaders."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _unit_tensor(value):
    array = np.asarray(value)
    tensor = torch.as_tensor(np.array(array, copy=True))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255)
    else:
        tensor = tensor.float()
    return tensor


def image_to_rgb_tensor(value):
    tensor = _unit_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f'Expected 2D image with optional channel, got {tuple(tensor.shape)}')
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] not in (1, 3):
        raise ValueError(f'Cannot identify image channel axis: {tuple(tensor.shape)}')
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    return tensor.contiguous()


def volume_to_rgb_tensor(value):
    tensor = _unit_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 4:
        raise ValueError(f'Expected 3D volume with optional channel, got {tuple(tensor.shape)}')
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(3, 0, 1, 2)
    elif tensor.shape[0] not in (1, 3):
        raise ValueError(f'Cannot identify volume channel axis: {tuple(tensor.shape)}')
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1, -1)
    return tensor.contiguous()


class CyclingLoaders:
    def __init__(self, loaders):
        if not loaders:
            raise ValueError('At least one dataset loader is required')
        self.loaders = dict(loaders)
        self.iterators = {name: iter(loader) for name, loader in self.loaders.items()}

    def next(self, dataset):
        if dataset not in self.loaders:
            raise KeyError(f'Unknown loader: {dataset}')
        try:
            return next(self.iterators[dataset])
        except StopIteration:
            self.iterators[dataset] = iter(self.loaders[dataset])
            return next(self.iterators[dataset])


def build_train_loaders(specs, root, batch_size_2d=128, batch_size_3d=32, download=False):
    import medmnist

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    loaders = {}
    counts = {}
    for name, spec in specs.items():
        info = medmnist.INFO[name]
        dataset_class = getattr(medmnist, info['python_class'])
        transform = image_to_rgb_tensor if spec.dimension == '2d' else volume_to_rgb_tensor
        dataset = dataset_class(
            split='train', root=str(root), size=28, download=download,
            as_rgb=False, transform=transform,
        )
        batch_size = batch_size_2d if spec.dimension == '2d' else batch_size_3d
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        counts[name] = len(dataset)
    return loaders, counts
