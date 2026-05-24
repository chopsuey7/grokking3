"""
Shared utilities: plotting, Fourier transforms, logging.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict
import json


# ---------- Plotting configuration ----------

def setup_plotting():
    """Set up publication-quality matplotlib defaults."""
    plt.rcParams.update({
        'figure.figsize': (12, 7),
        'figure.dpi': 150,
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ---------- Discrete Fourier Transform ----------

def fourier_basis(p: int, device: str = 'cpu') -> torch.Tensor:
    """
    Construct the 1D DFT basis matrix for p elements.

    Returns: (p, p) real matrix where:
        Row 0: constant (1/sqrt(p))
        Row 2k-1: cos(2πk·n/p) * sqrt(2/p)  for k = 1, ..., (p-1)/2
        Row 2k:   sin(2πk·n/p) * sqrt(2/p)  for k = 1, ..., (p-1)/2
    """
    basis = torch.zeros(p, p, device=device)
    basis[0] = 1.0 / np.sqrt(p)

    n = torch.arange(p, device=device, dtype=torch.float32)
    for k in range(1, (p + 1) // 2):
        angle = 2 * np.pi * k * n / p
        basis[2 * k - 1] = torch.cos(angle) * np.sqrt(2.0 / p)
        basis[2 * k] = torch.sin(angle) * np.sqrt(2.0 / p)

    # If p is even, the last row is the Nyquist frequency
    if p % 2 == 0:
        n = torch.arange(p, device=device, dtype=torch.float32)
        basis[p - 1] = torch.cos(np.pi * n) / np.sqrt(p)

    return basis


def fourier_transform(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    Apply the 1D DFT along the first dimension of x.

    Args:
        x: (p, ...) tensor
        p: prime modulus (size of first dimension)

    Returns: (p, ...) Fourier-transformed tensor
    """
    basis = fourier_basis(p, device=x.device)  # (p, p)
    # x: (p, d) -> basis @ x: (p, d)
    return basis @ x


def compute_fourier_power(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    Compute the power spectrum per frequency from the DFT of x.

    Args:
        x: (p, d) tensor
        p: prime modulus

    Returns: (n_freq,) power per frequency, where n_freq = (p+1)//2
    """
    ft = fourier_transform(x, p)  # (p, d)

    # Group cos/sin pairs
    n_freq = (p + 1) // 2  # number of frequencies (including DC)
    power = torch.zeros(n_freq, device=x.device)

    # DC component (frequency 0)
    power[0] = (ft[0] ** 2).sum()

    # Frequency k: rows 2k-1 (cos) and 2k (sin)
    for k in range(1, n_freq):
        cos_idx = 2 * k - 1
        sin_idx = 2 * k
        if cos_idx < p:
            power[k] += (ft[cos_idx] ** 2).sum()
        if sin_idx < p:
            power[k] += (ft[sin_idx] ** 2).sum()

    return power


def get_key_frequencies(embedding: torch.Tensor, p: int, threshold: float = 0.1) -> List[int]:
    """
    Identify key Fourier frequencies from the embedding matrix.

    Args:
        embedding: (p, d_model) — first p rows of W_E (excluding = token)
        p: prime modulus
        threshold: fraction of max power to be considered 'key'

    Returns: sorted list of key frequency indices
    """
    power = compute_fourier_power(embedding, p)
    max_power = power.max()
    key_freqs = (power > threshold * max_power).nonzero(as_tuple=True)[0].tolist()
    # Remove DC component (freq 0) unless it's significant
    if 0 in key_freqs and power[0] < 0.5 * max_power:
        key_freqs.remove(0)
    return sorted(key_freqs)


# ---------- Loss computation with Fourier ablation ----------

def compute_restricted_loss(model, tokens, labels, key_freqs: List[int], p: int):
    """
    Compute loss using only the key Fourier frequencies in the logits.
    This measures how well the generalizing circuit alone performs.
    """
    logits, cache = model(tokens, return_cache=True)
    logits_ft = fourier_transform_2d_logits(logits, p)

    # Zero out non-key frequencies
    n_freq = (p + 1) // 2
    mask = torch.zeros(p, device=logits.device)
    for k in key_freqs:
        if k == 0:
            mask[0] = 1.0
        else:
            cos_idx = 2 * k - 1
            sin_idx = 2 * k
            if cos_idx < p:
                mask[cos_idx] = 1.0
            if sin_idx < p:
                mask[sin_idx] = 1.0

    # Apply mask in Fourier space and invert
    basis = fourier_basis(p, device=logits.device)
    logits_fourier = basis @ logits.T  # (p, batch)
    logits_fourier = logits_fourier * mask[:, None]
    logits_restricted = (basis.T @ logits_fourier).T  # (batch, p)

    loss = torch.nn.functional.cross_entropy(logits_restricted, labels)
    return loss.item()


def compute_excluded_loss(model, tokens, labels, key_freqs: List[int], p: int):
    """
    Compute loss using only the NON-key frequencies.
    This measures how well the memorization circuit alone performs.
    """
    logits, cache = model(tokens, return_cache=True)

    # Create mask for non-key frequencies
    mask = torch.ones(p, device=logits.device)
    for k in key_freqs:
        if k == 0:
            mask[0] = 0.0
        else:
            cos_idx = 2 * k - 1
            sin_idx = 2 * k
            if cos_idx < p:
                mask[cos_idx] = 0.0
            if sin_idx < p:
                mask[sin_idx] = 0.0

    basis = fourier_basis(p, device=logits.device)
    logits_fourier = basis @ logits.T
    logits_fourier = logits_fourier * mask[:, None]
    logits_excluded = (basis.T @ logits_fourier).T

    loss = torch.nn.functional.cross_entropy(logits_excluded, labels)
    return loss.item()


def fourier_transform_2d_logits(logits, p):
    """Helper for 2D Fourier transform on logits."""
    return logits  # placeholder — the actual transform is done in-function above


# ---------- Metrics ----------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------- Logging ----------

def save_metrics(metrics: Dict, filepath: str):
    """Save training metrics to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            clean[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            clean[k] = v
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)


def load_metrics(filepath: str) -> Dict:
    """Load training metrics from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)
