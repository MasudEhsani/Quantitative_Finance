"""
A minimal linear autoencoder, implemented in plain NumPy.

This exists to reproduce, without a TensorFlow/Keras dependency, the course
project's side-by-side comparison of "Absorption Ratio via PCA" vs.
"Absorption Ratio via Auto-Encoder". A single-hidden-layer autoencoder with
no bias terms and no activation functions, trained to minimize mean-squared
reconstruction error, recovers (up to an arbitrary rotation) the same
subspace as PCA restricted to the leading components -- this is the
classical Baldi & Hornik (1989) result. So a small NumPy implementation is
both dependency-light and a faithful stand-in for the original TensorFlow
``LinearAutoEncoder`` class, whose ``tf.contrib`` API no longer exists in
modern TensorFlow.

The encoder/decoder are trained with full-batch Adam directly on a window
of (standardized) asset returns -- rather than on covariance matrices as
"images", which was the original notebook's approach -- since that is the
more standard formulation and makes the PCA-equivalence exact.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearAutoencoderResult:
    W_encode: np.ndarray  # (n_assets, n_components)
    W_decode: np.ndarray  # (n_components, n_assets)
    loss_history: np.ndarray
    explained_variance_ratio: float


def _adam_update(param, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    m[:] = beta1 * m + (1 - beta1) * grad
    v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_linear_autoencoder(
    X: np.ndarray,
    n_components: int,
    epochs: int = 300,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> LinearAutoencoderResult:
    """Train a linear (no bias, no activation) autoencoder via full-batch Adam.

    Arguments:
        X -- np.ndarray, shape (n_samples, n_assets); should already be
            centered (mean 0 per column) for the explained-variance
            calculation to be meaningful
        n_components -- bottleneck width (analogous to number of PCA
            components retained)
        epochs -- number of full-batch gradient steps
        learning_rate -- Adam learning rate
        seed -- RNG seed for weight initialization

    Return:
        LinearAutoencoderResult with the learned weights, the loss curve,
        and the fraction of variance explained by the reconstruction.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_assets = X.shape

    W1 = rng.normal(scale=0.01, size=(n_assets, n_components))
    W2 = rng.normal(scale=0.01, size=(n_components, n_assets))
    m1, v1 = np.zeros_like(W1), np.zeros_like(W1)
    m2, v2 = np.zeros_like(W2), np.zeros_like(W2)

    loss_history = np.zeros(epochs)
    total_var = np.sum((X - X.mean(axis=0)) ** 2)

    for epoch in range(1, epochs + 1):
        codes = X @ W1
        recon = codes @ W2
        error = recon - X

        loss_history[epoch - 1] = np.mean(error ** 2)

        grad_recon = (2.0 / (n_samples * n_assets)) * error
        grad_W2 = codes.T @ grad_recon
        grad_codes = grad_recon @ W2.T
        grad_W1 = X.T @ grad_codes

        _adam_update(W1, grad_W1, m1, v1, epoch, lr=learning_rate)
        _adam_update(W2, grad_W2, m2, v2, epoch, lr=learning_rate)

    recon = (X @ W1) @ W2
    explained_variance_ratio = 1.0 - np.sum((X - recon) ** 2) / total_var

    return LinearAutoencoderResult(
        W_encode=W1,
        W_decode=W2,
        loss_history=loss_history,
        explained_variance_ratio=float(explained_variance_ratio),
    )


def autoencoder_absorption_ratio(X: np.ndarray, n_components: int, **train_kwargs) -> float:
    """Convenience wrapper: train a linear autoencoder and return the
    fraction of variance its bottleneck reconstructs -- the autoencoder
    analogue of ``pca_analysis.absorption_ratio``.

    Arguments:
        X -- np.ndarray, shape (n_samples, n_assets), centered returns
        n_components -- bottleneck width
        **train_kwargs -- forwarded to ``train_linear_autoencoder``

    Return:
        explained_variance_ratio -- float in roughly [0, 1]
    """
    result = train_linear_autoencoder(X, n_components, **train_kwargs)
    return result.explained_variance_ratio
