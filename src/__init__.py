"""Dimension-aware joint-learning components for the MedMNIST benchmark."""

from .models import DatasetSpec, DimensionAwareModel, select_specs, specs_from_plan

__all__ = ['DatasetSpec', 'DimensionAwareModel', 'select_specs', 'specs_from_plan']
