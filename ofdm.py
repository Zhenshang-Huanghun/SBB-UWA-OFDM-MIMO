from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OFDMConfig:
    K: int = 512
    CP: int = 128
    P: int = 256
    Nt: int = 2
    Nr: int = 2

    def __post_init__(self):
        if min(self.K, self.P, self.Nt, self.Nr) < 1 or not 0 <= self.CP <= self.K:
            raise ValueError("Invalid OFDM dimensions.")
        if self.P > self.K or self.P % self.Nt or self.K % (self.P // self.Nt):
            raise ValueError("P/Nt and K/(P/Nt) must be integers, with P <= K.")

    @property
    def pilot_indices(self):
        count = self.P // self.Nt
        return np.arange(self.Nt)[:, None] + np.arange(count) * (self.K // count)

    @property
    def data_indices(self):
        return np.setdiff1d(np.arange(self.K), self.pilot_indices.ravel())


def qpsk_modulate(bits):
    """Map bits (..., 2M) to unnormalized QPSK symbols (..., M)."""
    bits = np.asarray(bits, dtype=np.int8)
    pairs = bits.reshape(*bits.shape[:-1], -1, 2)
    return (2 * pairs[..., 0] - 1) + 1j * (2 * pairs[..., 1] - 1)


def qpsk_demodulate(symbols):
    """Return nearest-neighbor QPSK decisions in transmitter-major order."""
    symbols = np.asarray(symbols)
    constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    labels = np.array([[1, 1], [0, 1], [0, 0], [1, 0]], dtype=np.int8)
    nearest = np.abs(symbols[..., None] - constellation).argmin(axis=-1)
    return labels[nearest].reshape(*symbols.shape[:-1], -1)


def pilot_symbols(cfg, rng):
    """Create known pilots (Nt, P/Nt) from a caller-supplied RNG."""
    bits = rng.integers(0, 2, size=(cfg.Nt, 2 * (cfg.P // cfg.Nt)))
    return qpsk_modulate(bits)


def map_subcarriers(cfg, data_bits, pilots):
    """Map data bits (..., Nt, 2(K-P)) and pilots (Nt, P/Nt)."""
    data_bits = np.asarray(data_bits)
    pilots = np.asarray(pilots)
    if data_bits.shape[-2:] != (cfg.Nt, 2 * (cfg.K - cfg.P)):
        raise ValueError("data_bits must end in (Nt, 2(K-P)).")
    if pilots.shape != (cfg.Nt, cfg.P // cfg.Nt) or np.any(pilots == 0):
        raise ValueError("pilots must be nonzero and have shape (Nt, P/Nt).")
    x = np.zeros((*data_bits.shape[:-1], cfg.K), dtype=np.complex128)
    x[..., cfg.data_indices] = qpsk_modulate(data_bits)
    for j, indices in enumerate(cfg.pilot_indices):
        x[..., j, indices] = pilots[j]
    return x


def ofdm_modulate(cfg, frequency_symbols):
    """Unitary IFFT and cyclic prefix along the last axis."""
    symbols = np.fft.ifft(frequency_symbols, axis=-1, norm="ortho")
    if cfg.CP == 0:
        return symbols
    return np.concatenate((symbols[..., -cfg.CP:], symbols), axis=-1)


def ofdm_demodulate(cfg, received_samples):
    """Remove the prefix and apply a unitary FFT; input ends in (Nr, K+CP)."""
    samples = np.asarray(received_samples)[..., cfg.CP:cfg.CP + cfg.K]
    if samples.shape[-1] != cfg.K:
        raise ValueError("Insufficient samples for one OFDM symbol.")
    return np.fft.fft(samples, axis=-1, norm="ortho")


def transmit(cfg, data_bits, pilots):
    """Return time-domain symbols and their mapped frequency-domain symbols."""
    frequency = map_subcarriers(cfg, data_bits, pilots)
    return ofdm_modulate(cfg, frequency), frequency


def zf_detect(estimate, received, data_indices):
    """Detect data symbols (Nt, M) from estimate (Nr, Nt, K) and received (Nr, K)."""
    h = np.moveaxis(np.asarray(estimate)[..., data_indices], -1, 0)
    y = np.asarray(received)[:, data_indices].T
    return (np.linalg.pinv(h) @ y[..., None])[..., 0].T


def nmse(reference, estimate):
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    return float(np.sum(np.abs(estimate - reference) ** 2) /
                 (np.sum(np.abs(reference) ** 2) + 1e-12))


def bit_errors(bits, detected_symbols):
    decisions = qpsk_demodulate(detected_symbols).ravel()
    bits = np.asarray(bits).ravel()
    return int(np.count_nonzero(bits != decisions)), bits.size
