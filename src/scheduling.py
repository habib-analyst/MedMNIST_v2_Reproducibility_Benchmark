"""Dataset-balanced scheduling utilities."""


def temperature_probabilities(sample_counts, temperature=0.0):
    if not sample_counts or any(value <= 0 for value in sample_counts.values()):
        raise ValueError('Every dataset must have a positive sample count')
    if not 0 <= temperature <= 1:
        raise ValueError('temperature must be between zero and one')
    weights = {name: float(count) ** temperature for name, count in sample_counts.items()}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}
