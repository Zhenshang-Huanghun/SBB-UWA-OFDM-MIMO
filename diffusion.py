import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _extract(values, steps, x):
    return values[steps.to(values.device)].to(x).reshape(-1, *([1] * (x.ndim - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, total_steps=400, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if total_steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("Invalid diffusion schedule.")
        self.total_steps = int(total_steps)
        betas = torch.linspace(beta_start, beta_end, total_steps, dtype=torch.float64)
        alphas = 1 - betas
        alpha_bar = alphas.cumprod(0)
        previous = F.pad(alpha_bar, (1, 0), value=1)[:-1]
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())
        self.register_buffer("coeff1", alphas.rsqrt())
        self.register_buffer("coeff2", alphas.rsqrt() * betas / (1 - alpha_bar).sqrt())
        self.register_buffer("posterior_variance", betas * (1 - previous) / (1 - alpha_bar))

    def _steps(self, step, x):
        steps = torch.as_tensor(step, dtype=torch.long, device=x.device)
        if steps.ndim == 0:
            steps = steps.expand(x.shape[0])
        if steps.shape != (x.shape[0],) or torch.any((steps < 0) | (steps >= self.total_steps)):
            raise ValueError("Expected one valid diffusion index per sample.")
        return steps

    def q_sample(self, x0, t, noise=None):
        t = self._steps(t, x0)
        noise = torch.randn_like(x0) if noise is None else noise
        state = (_extract(self.sqrt_alpha_bar, t, x0) * x0 +
                 _extract(self.sqrt_one_minus_alpha_bar, t, x0) * noise)
        return state, noise

    def loss(self, model, x0, t=None, noise=None):
        if t is None:
            t = torch.randint(self.total_steps, (len(x0),), device=x0.device)
        t = self._steps(t, x0)
        state, noise = self.q_sample(x0, t, noise)
        return F.mse_loss(model(state, t), noise)

    def initialize(self, h_ls, start_step):
        steps = self._steps(start_step, h_ls)
        return _extract(self.sqrt_alpha_bar, steps, h_ls) * h_ls

    @torch.no_grad()
    def sample_ddpm(self, model, h_ls, start_step, stochastic=True):
        steps = self._steps(start_step, h_ls)
        current = self.initialize(h_ls, steps)
        for step in range(int(steps.max()), -1, -1):
            t = torch.full_like(steps, step)
            mean = (_extract(self.coeff1, t, current) * current -
                    _extract(self.coeff2, t, current) * model(current, t))
            if stochastic and step > 0:
                mean = mean + _extract(self.posterior_variance, t, current).sqrt() * torch.randn_like(current)
            active = (steps >= step).reshape(-1, *([1] * (current.ndim - 1)))
            current = torch.where(active, mean, current)
        return current

    @torch.no_grad()
    def sample_ddim(self, model, h_ls, start_step):
        """One-step deterministic prediction using the DDPM noise estimator."""
        steps = self._steps(start_step, h_ls)
        state = self.initialize(h_ls, steps)
        noise = model(state, steps)
        return ((state - _extract(self.sqrt_one_minus_alpha_bar, steps, state) * noise) /
                _extract(self.sqrt_alpha_bar, steps, state))

    def start_steps_from_nmse(self, nmse, t_min=1):
        values = np.asarray(nmse, dtype=np.float64)
        if np.any(values <= 0) or not np.isfinite(values).all() or not 0 <= t_min < self.total_steps:
            raise ValueError("NMSE must be positive and finite, and t_min within the schedule.")
        alpha_bar = self.alpha_bar.detach().cpu().numpy()
        schedule_snr = 10 * np.log10(alpha_bar / (1 - alpha_bar))
        target = np.clip(-10 * np.log10(values), schedule_snr[-1], schedule_snr[0])
        indices = np.abs(target.reshape(-1, 1) - schedule_snr).argmin(axis=1)
        indices = np.clip(indices, t_min, self.total_steps - 1).reshape(values.shape)
        return int(indices) if values.ndim == 0 else indices

    def start_steps_from_snr(self, snr_db, table_snr, table_nmse, t_min=1):
        table_snr = np.asarray(table_snr, dtype=float)
        table_nmse = np.asarray(table_nmse, dtype=float)
        if table_snr.ndim != 1 or table_nmse.shape != table_snr.shape or len(table_snr) < 2:
            raise ValueError("Expected equally sized one-dimensional calibration arrays.")
        order = table_snr.argsort()
        nmse = np.interp(snr_db, table_snr[order], table_nmse[order])
        return self.start_steps_from_nmse(nmse, t_min)
