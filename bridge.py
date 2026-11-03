import torch
import torch.nn.functional as F
from torch import nn


class BrownianBridgeProcess(nn.Module):
    def __init__(self, total_steps=100, max_var=1.0):
        super().__init__()
        if total_steps < 2 or max_var < 0:
            raise ValueError("Invalid bridge schedule.")
        self.total_steps = int(total_steps)
        self.max_var = float(max_var)
        m = (torch.arange(total_steps, dtype=torch.float64) + 1) / total_steps
        self.register_buffer("m", m.float())
        self.register_buffer("delta", (2 * max_var * (m - m.square())).float())

    def q_sample(self, h, h_ls, t, noise=None):
        """Sample a bridge state; network indices run from 0 to T-1."""
        noise = torch.randn_like(h) if noise is None else noise
        shape = (-1, *([1] * (h.ndim - 1)))
        m = self.m[t.to(self.m.device)].to(h).reshape(shape)
        delta = self.delta[t.to(self.delta.device)].to(h).reshape(shape)
        return (1 - m) * h + m * h_ls + delta.sqrt() * noise


@torch.no_grad()
def endpoint_estimate(model, h_ls, total_steps=None):
    steps = model.total_steps if total_steps is None else total_steps
    t = torch.full((len(h_ls),), steps - 1, dtype=torch.long, device=h_ls.device)
    return model(h_ls, h_ls, t)


def delay_group_l1(pred_phys, cir_length=128, epsilon=1e-12):
    """Group norm over links, averaged over the first L delay taps and the batch."""
    real, imag = pred_phys.chunk(2, dim=-2)
    taps = torch.fft.ifft(torch.complex(real, imag), dim=-1)[..., :cir_length]
    return (taps.abs().square().sum(dim=-2) + epsilon).sqrt().mean()


def bridge_loss(pred, target, normalization, delay_lambda=0.0, cir_length=128):
    loss = F.mse_loss(pred, target)
    if delay_lambda:
        loss = loss + delay_lambda * delay_group_l1(normalization.decode(pred), cir_length)
    return loss
