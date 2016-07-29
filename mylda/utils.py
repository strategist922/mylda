import numpy as np


def n2s(counts):
    """Convert a counts vector to corresponding samples."""
    samples = []
    for (value, count) in enumerate(counts):
        samples += [value, ] * count
    return np.random.permutation(samples)
