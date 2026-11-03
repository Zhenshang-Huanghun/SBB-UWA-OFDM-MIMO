from dataclasses import dataclass

import numpy as np


def pilot_observations(cfg, received, transmitted):
    """Return pilot LS observations with shape (Nr, Nt, P/Nt)."""
    z = np.empty((cfg.Nr, cfg.Nt, cfg.P // cfg.Nt), dtype=np.complex128)
    for j, indices in enumerate(cfg.pilot_indices):
        values = transmitted[j, indices]
        if np.any(values == 0):
            raise ValueError("Pilot symbols must be nonzero.")
        z[:, j] = received[:, indices] / values
    return z


def estimate_ls(cfg, received, transmitted):
    """Interpolate pilot LS observations to a complex (Nr, Nt, K) array."""
    z = pilot_observations(cfg, received, transmitted)
    estimate = np.empty((cfg.Nr, cfg.Nt, cfg.K), dtype=np.complex128)
    for j, indices in enumerate(cfg.pilot_indices):
        for i in range(cfg.Nr):
            estimate[i, j] = np.interp(np.arange(cfg.K), indices, z[i, j].real)
            estimate[i, j] += 1j * np.interp(np.arange(cfg.K), indices, z[i, j].imag)
    return estimate


@dataclass(frozen=True)
class LMMSEStatistics:
    mean: np.ndarray
    covariance: np.ndarray


def fit_lmmse(training_responses):
    """Fit per-link moments from training responses (N, Nr, Nt, K)."""
    values = np.asarray(training_responses, dtype=np.complex128)
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("Expected a nonempty (N, Nr, Nt, K) array.")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = np.einsum("nrtk,nrtl->rtkl", centered, centered.conj()) / len(values)
    covariance = (covariance + covariance.swapaxes(-1, -2).conj()) * 0.5
    return LMMSEStatistics(mean, covariance)


def estimate_lmmse(cfg, received, transmitted, statistics, noise_variance,
                   diagonal_loading=1e-8):
    """Affine frequency-domain LMMSE; noise_variance is the complex FFT noise power."""
    z = pilot_observations(cfg, received, transmitted)
    estimate = np.empty((cfg.Nr, cfg.Nt, cfg.K), dtype=np.complex128)
    for j, indices in enumerate(cfg.pilot_indices):
        variance = noise_variance / np.abs(transmitted[j, indices]) ** 2
        for i in range(cfg.Nr):
            mean = statistics.mean[i, j]
            covariance = statistics.covariance[i, j]
            c_pp = covariance[np.ix_(indices, indices)]
            scale = max(np.trace(c_pp).real / len(indices), np.finfo(float).eps)
            system = c_pp + np.diag(variance + diagonal_loading * scale)
            weights = np.linalg.solve(system, z[i, j] - mean[indices])
            estimate[i, j] = mean + covariance[:, indices] @ weights
    return estimate
