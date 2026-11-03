from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from bridge import BrownianBridgeProcess, bridge_loss, endpoint_estimate
from diffusion import GaussianDiffusion
from models import BridgeDenoiser, DDPMUNet
from representation import Normalization, complex_to_channels


@dataclass
class TrainingResult:
    model: torch.nn.Module
    process: torch.nn.Module
    normalization: Normalization
    history: list
    best_epoch: int


def _channels(values):
    values = torch.as_tensor(values)
    values = complex_to_channels(values) if values.is_complex() else values.float()
    if values.ndim != 3 or len(values) == 0 or values.shape[1] % 2 or values.shape[2] % 4:
        raise ValueError("Expected nonempty (N, 2NrNt, K) data with K divisible by 4.")
    return values.detach().cpu()


def _prepare_device(device, seed, epochs):
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _loader(*values, batch_size, shuffle):
    return DataLoader(TensorDataset(*values), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def _state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def _epsilon_validation(model, process, loader, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    loss_sum = 0.0
    for (x,) in loader:
        x = x.to(device)
        t = torch.randint(process.total_steps, (len(x),), device=device, generator=generator)
        noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
        loss_sum += float(process.loss(model, x, t=t, noise=noise)) * len(x)
    return loss_sum / len(loader.dataset)


@torch.no_grad()
def _endpoint_validation(model, process, loader, normalization, device):
    model.eval()
    total = 0.0
    for endpoint, true in loader:
        endpoint, true = endpoint.to(device), true.to(device)
        prediction = endpoint_estimate(model, endpoint, process.total_steps)
        prediction, true = normalization.decode(prediction), normalization.decode(true)
        error = (prediction - true).square().flatten(1).sum(1)
        energy = true.square().flatten(1).sum(1)
        total += float((error / (energy + 1e-12)).sum())
    return total / len(loader.dataset)


def train_diffusion(h_train, h_val, *, epochs=1000, batch_size=64, lr=2e-4,
                    weight_decay=0.05, grad_clip=1.0, total_steps=400,
                    beta_start=1e-4, beta_end=0.02, base_channels=32,
                    seed=2026, device=None, validation=None, verbose=True):
    """Train epsilon prediction; optional validation(model, process, normalization) returns a score."""
    device = _prepare_device(device, seed, epochs)
    h_train, h_val = _channels(h_train), _channels(h_val)
    if h_train.shape[1:] != h_val.shape[1:]:
        raise ValueError("Training and validation feature shapes must agree.")
    normalization = Normalization.fit(h_train)
    train_loader = _loader(normalization.encode(h_train), batch_size=batch_size, shuffle=True)
    val_loader = _loader(normalization.encode(h_val), batch_size=batch_size, shuffle=False)
    model = DDPMUNet(total_steps=total_steps, in_channels=h_train.shape[1],
                     out_channels=h_train.shape[1], base_channels=base_channels).to(device)
    process = GaussianDiffusion(total_steps=total_steps, beta_start=beta_start,
                                beta_end=beta_end).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.3, div_factor=10.0, final_div_factor=100.0, anneal_strategy="cos")
    history, best_score, best_state, best_epoch = [], float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = process.loss(model, x)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(x)
        val_loss = _epsilon_validation(model, process, val_loader, device, seed + 1)
        with torch.no_grad():
            score = val_loss if validation is None else float(validation(model, process, normalization))
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset),
                  "val_epsilon_mse": val_loss, "selection_score": score}
        history.append(record)
        if score < best_score:
            best_score, best_state, best_epoch = score, _state(model), epoch
        if verbose:
            print(f"epoch {epoch}: train={record['train_loss']:.6e}, val={score:.6e}")
    if best_state is None:
        raise FloatingPointError("Validation did not produce a finite best score.")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model, process, normalization, history, best_epoch)


def train_bridge(h_ls_train, h_true_train, h_ls_val, h_true_val, *, loss="l2",
                 epochs=300, batch_size=64, lr=2e-4, weight_decay=0.05,
                 grad_clip=1.0, total_steps=100, max_var=1.0, base_channels=32,
                 delay_lambda=0.2, cir_length=128, val_limit=256,
                 seed=2026, device=None, verbose=True):
    """Train paired channel reconstruction from physical complex or packed real arrays."""
    if loss not in ("l2", "l1"):
        raise ValueError("loss must be l2 or l1.")
    device = _prepare_device(device, seed, epochs)
    h_ls_train, h_true_train, h_ls_val, h_true_val = map(
        _channels, (h_ls_train, h_true_train, h_ls_val, h_true_val))
    if h_ls_train.shape != h_true_train.shape or h_ls_val.shape != h_true_val.shape:
        raise ValueError("Each endpoint array must match its target array.")
    if h_true_train.shape[1:] != h_true_val.shape[1:]:
        raise ValueError("Training and validation feature shapes must agree.")
    normalization = Normalization.fit(h_true_train)
    train_loader = _loader(normalization.encode(h_ls_train), normalization.encode(h_true_train),
                           batch_size=batch_size, shuffle=True)
    if val_limit is not None:
        if val_limit < 1:
            raise ValueError("val_limit must be positive or None.")
        h_ls_val, h_true_val = h_ls_val[:val_limit], h_true_val[:val_limit]
    val_loader = _loader(normalization.encode(h_ls_val), normalization.encode(h_true_val),
                         batch_size=batch_size, shuffle=False)
    model = BridgeDenoiser(total_steps=total_steps, channels=h_true_train.shape[1],
                           base_channels=base_channels).to(device)
    process = BrownianBridgeProcess(total_steps=total_steps, max_var=max_var).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    regularization = delay_lambda if loss == "l1" else 0.0
    history, best_score, best_state, best_epoch = [], float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for endpoint, true in train_loader:
            endpoint, true = endpoint.to(device), true.to(device)
            t = torch.randint(total_steps, (len(true),), device=device)
            state = process.q_sample(true, endpoint, t)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(state, endpoint, t)
            objective = bridge_loss(prediction, true, normalization,
                                    delay_lambda=regularization, cir_length=cir_length)
            objective.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += float(objective.detach()) * len(true)
        scheduler.step()
        val_nmse = _endpoint_validation(model, process, val_loader, normalization, device)
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset),
                  "val_nmse": val_nmse}
        history.append(record)
        if val_nmse < best_score:
            best_score, best_state, best_epoch = val_nmse, _state(model), epoch
        if verbose:
            print(f"epoch {epoch}: train={record['train_loss']:.6e}, val_nmse={val_nmse:.6e}")
    if best_state is None:
        raise FloatingPointError("Validation did not produce a finite best score.")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model, process, normalization, history, best_epoch)


train_ddpm = train_diffusion
train_ddim = train_diffusion


def train_bridge_l2(h_ls_train, h_true_train, h_ls_val, h_true_val, **kwargs):
    return train_bridge(h_ls_train, h_true_train, h_ls_val, h_true_val, loss="l2", **kwargs)


def train_bridge_l1(h_ls_train, h_true_train, h_ls_val, h_true_val, **kwargs):
    return train_bridge(h_ls_train, h_true_train, h_ls_val, h_true_val, loss="l1", **kwargs)
