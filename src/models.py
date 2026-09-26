"""Dataset-conditioned 2D, 3D and unified MedMNIST models."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dimension: str
    task: str
    classes: int

    def __post_init__(self):
        if self.dimension not in {'2d', '3d'}:
            raise ValueError(f'Unsupported dimension: {self.dimension}')
        if self.classes < 2:
            raise ValueError('Each dataset requires at least two outputs')


def specs_from_plan(plan):
    """Build the native label-space registry from the audited experiment plan."""
    return {
        item['flag']: DatasetSpec(item['flag'], item['dimension'], item['task'], int(item['classes']))
        for item in plan['datasets']
    }


def select_specs(specs, mode, requested=None):
    if mode not in {'2d', '3d', 'unified'}:
        raise ValueError(f'Unsupported mode: {mode}')
    names = list(requested) if requested else [
        name for name, spec in specs.items() if mode == 'unified' or spec.dimension == mode
    ]
    missing = [name for name in names if name not in specs]
    if missing:
        raise ValueError(f'Unknown requested datasets: {missing}')
    selected = {name: specs[name] for name in names}
    wrong = [name for name, spec in selected.items() if mode != 'unified' and spec.dimension != mode]
    if wrong:
        raise ValueError(f'Dataset does not belong to {mode} mode: {wrong}')
    if not selected:
        raise ValueError('At least one dataset must be selected')
    return selected


class Stem2D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden = max(16, embed_dim // 2)
        self.network = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, embed_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, value):
        features = self.network(value)
        return features.flatten(2).transpose(1, 2)


class Stem3D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden = max(16, embed_dim // 2)
        self.network = nn.Sequential(
            nn.Conv3d(3, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv3d(hidden, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, embed_dim), nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )

    def forward(self, value):
        features = self.network(value)
        return features.flatten(2).transpose(1, 2)


class DimensionAwareModel(nn.Module):
    """Route native 2D/3D inputs through shared conditioned representation blocks."""
    def __init__(self, specs, mode='unified', embed_dim=128, depth=2, heads=4, dropout=0.0):
        super().__init__()
        if mode not in {'2d', '3d', 'unified'}:
            raise ValueError(f'Unsupported mode: {mode}')
        if embed_dim % heads:
            raise ValueError('embed_dim must be divisible by attention heads')
        self.mode = mode
        self.specs = {
            name: spec for name, spec in specs.items()
            if mode == 'unified' or spec.dimension == mode
        }
        if not self.specs:
            raise ValueError(f'No datasets are available for mode {mode}')
        self.dataset_names = tuple(sorted(self.specs))
        self.dataset_index = {name: index for index, name in enumerate(self.dataset_names)}
        self.stems = nn.ModuleDict()
        if mode in {'2d', 'unified'}:
            self.stems['2d'] = Stem2D(embed_dim)
        if mode in {'3d', 'unified'}:
            self.stems['3d'] = Stem3D(embed_dim)
        self.dataset_embeddings = nn.Embedding(len(self.dataset_names), embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False,
        )
        self.shared_backbone = nn.TransformerEncoder(layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.heads = nn.ModuleDict({name: nn.Linear(embed_dim, spec.classes) for name, spec in self.specs.items()})

    def forward(self, inputs, dataset):
        if dataset not in self.specs:
            raise KeyError(f'Unknown dataset for mode {self.mode}: {dataset}')
        spec = self.specs[dataset]
        expected_rank = 4 if spec.dimension == '2d' else 5
        if inputs.ndim != expected_rank:
            raise ValueError(f'{dataset} expects {spec.dimension} input rank {expected_rank}, got {inputs.ndim}')
        tokens = self.stems[spec.dimension](inputs)
        index = torch.tensor(self.dataset_index[dataset], device=inputs.device)
        task_token = self.dataset_embeddings(index).reshape(1, 1, -1).expand(inputs.shape[0], -1, -1)
        encoded = self.shared_backbone(torch.cat([task_token, tokens], dim=1))
        return self.heads[dataset](self.final_norm(encoded[:, 0]))
