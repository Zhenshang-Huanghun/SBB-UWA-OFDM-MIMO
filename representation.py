from dataclasses import dataclass

import torch


def complex_to_channels(values):
    """Pack (..., Nr, Nt, K) as (..., 2*Nr*Nt, K), real links then imaginary links."""
    values = torch.as_tensor(values)
    if not values.is_complex() or values.ndim < 3:
        raise ValueError("Expected a complex tensor ending in (Nr, Nt, K).")
    shape = (*values.shape[:-3], -1, values.shape[-1])
    return torch.cat((values.real.reshape(shape), values.imag.reshape(shape)), dim=-2).float()


def channels_to_complex(values, nr=2, nt=2):
    values = torch.as_tensor(values)
    if values.shape[-2] != 2 * nr * nt:
        raise ValueError("Feature count must be 2*nr*nt.")
    real, imag = values.chunk(2, dim=-2)
    return torch.complex(real, imag).reshape(*values.shape[:-2], nr, nt, values.shape[-1])


@dataclass(frozen=True)
class Normalization:
    mean: float
    std: float

    @classmethod
    def fit(cls, training_values):
        values = torch.as_tensor(training_values)
        if values.is_complex():
            raise ValueError("Pack complex values before fitting normalization.")
        values = values.float()
        return cls(float(values.mean()), max(float(values.std(unbiased=False)), 1e-8))

    def encode(self, values):
        return (values - self.mean) / self.std

    def decode(self, values):
        return values * self.std + self.mean
