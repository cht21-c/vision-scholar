"""Small executable CV kernels, intentionally readable and free of deep-learning dependencies."""
from __future__ import annotations

import numpy as np


def patchify(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Convert H×W×C image into N×(P²C) non-overlapping flattened patches."""
    height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    return (image.reshape(height // patch_size, patch_size, width // patch_size,
                          patch_size, channels)
            .transpose(0, 2, 1, 3, 4)
            .reshape(-1, patch_size * patch_size * channels))


def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray,
                                 value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute softmax(QKᵀ / sqrt(d_k)) V with a numerically stable softmax."""
    scores = query @ key.T / np.sqrt(query.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value, weights


def residual_block(x: np.ndarray, weight1: np.ndarray, weight2: np.ndarray) -> np.ndarray:
    """Teaching-only shape-preserving residual MLP: ReLU(x + W2 ReLU(W1 x))."""
    hidden = np.maximum(x @ weight1, 0)
    return np.maximum(x + hidden @ weight2, 0)
