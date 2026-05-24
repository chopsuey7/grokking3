"""
Component 1: Reproduce Grokking on Modular Addition

Trains a one-layer transformer on (a + b) mod p and produces
training/validation loss and accuracy curves showing grokking.

Usage:
    python -m src.train          # Run from grokking/ directory
    # Or paste into a Colab cell after uploading src/
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import OneLayerTransformer
from src.data import make_modular_addition_data
from src.utils import setup_plotting, compute_accuracy, save_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

P = 97                      # Prime modulus
D_MODEL = 128               # Model dimension
N_HEADS = 4                 # Number of attention heads
D_MLP = 512                 # MLP hidden dimension
TRAIN_FRAC = 0.3            # Fraction of data for training
LR = 1e-3                   # Learning rate
WEIGHT_DECAY = 1.0          # Weight decay (critical for grokking!)
BETAS = (0.9, 0.98)         # Adam betas
NUM_EPOCHS = 40_000         # Number of training epochs
LOG_EVERY = 100             # Log metrics every N epochs
CHECKPOINT_EVERY = 2000     # Save checkpoint every N epochs
SEED = 42                   # Random seed

FIGURES_DIR = Path("figures")
CHECKPOINTS_DIR = Path("checkpoints")


def train_model(
    p=P, d_model=D_MODEL, n_heads=N_HEADS, d_mlp=D_MLP,
    train_frac=TRAIN_FRAC, lr=LR, weight_decay=WEIGHT_DECAY,
    num_epochs=NUM_EPOCHS, log_every=LOG_EVERY,
    checkpoint_every=CHECKPOINT_EVERY, seed=SEED,
    device=None, save_checkpoints=True, return_model=True,
    figures_dir=None, checkpoints_dir=None,
    quiet=False
):
    """
    Train a one-layer transformer on modular addition and reproduce grokking.

    Returns:
        model: trained model (if return_model=True)
        metrics: dict with training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if figures_dir is None:
        figures_dir = FIGURES_DIR
    if checkpoints_dir is None:
        checkpoints_dir = CHECKPOINTS_DIR

    figures_dir = Path(figures_dir)
    checkpoints_dir = Path(checkpoints_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not quiet:
        print(f"Training on {device} | p={p} | train_frac={train_frac}")
        print(f"Architecture: d_model={d_model}, n_heads={n_heads}, d_mlp={d_mlp}")
        print(f"Optimizer: AdamW(lr={lr}, wd={weight_decay}, betas={BETAS})")
        print(f"Epochs: {num_epochs}")

    # Data
    train_tokens, train_labels, test_tokens, test_labels = make_modular_addition_data(
        p, train_frac=train_frac, seed=seed, device=device
    )
    if not quiet:
        print(f"Train: {len(train_tokens)} | Test: {len(test_tokens)}")

    # Model
    model = OneLayerTransformer(p, d_model, n_heads, d_mlp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if not quiet:
        print(f"Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=BETAS
    )

    # Metrics tracking
    metrics = {
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'weight_norm': [],
        'config': {
            'p': p, 'd_model': d_model, 'n_heads': n_heads, 'd_mlp': d_mlp,
            'train_frac': train_frac, 'lr': lr, 'weight_decay': weight_decay,
            'num_epochs': num_epochs, 'seed': seed
        }
    }

    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training", disable=quiet)
    for epoch in pbar:
        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        logits = model(train_tokens)
        train_loss = F.cross_entropy(logits, train_labels)
        train_loss.backward()
        optimizer.step()

        # --- Logging ---
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_tokens)
                test_logits = model(test_tokens)

                t_loss = F.cross_entropy(train_logits, train_labels).item()
                v_loss = F.cross_entropy(test_logits, test_labels).item()
                t_acc = compute_accuracy(train_logits, train_labels)
                v_acc = compute_accuracy(test_logits, test_labels)
                w_norm = model.sum_squared_weights().item()

            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(t_loss)
            metrics['test_loss'].append(v_loss)
            metrics['train_acc'].append(t_acc)
            metrics['test_acc'].append(v_acc)
            metrics['weight_norm'].append(w_norm)

            pbar.set_postfix({
                'train_loss': f'{t_loss:.4f}',
                'test_loss': f'{v_loss:.4f}',
                'train_acc': f'{t_acc:.2%}',
                'test_acc': f'{v_acc:.2%}'
            })

        # --- Checkpoints ---
        if save_checkpoints and (epoch % checkpoint_every == 0 or epoch == num_epochs - 1):
            ckpt_path = checkpoints_dir / f"model_epoch_{epoch:06d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss.item(),
                'config': metrics['config']
            }, ckpt_path)

    # Save metrics
    save_metrics(metrics, str(figures_dir / "training_metrics.json"))

    # Plot results
    plot_training_curves(metrics, figures_dir)

    if not quiet:
        print(f"\nFinal — Train acc: {metrics['train_acc'][-1]:.2%} | Test acc: {metrics['test_acc'][-1]:.2%}")
        print(f"Figures saved to {figures_dir}/")

    if return_model:
        return model, metrics
    return metrics


def plot_training_curves(metrics: dict, figures_dir: Path):
    """Plot training and validation loss/accuracy curves showing grokking."""
    setup_plotting()

    epochs = metrics['epochs']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss curves ---
    ax = axes[0]
    ax.semilogy(epochs, metrics['train_loss'], label='Train Loss', color='#2196F3', linewidth=2)
    ax.semilogy(epochs, metrics['test_loss'], label='Test Loss', color='#FF5722', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss (log scale)')
    ax.set_title('Training & Validation Loss')
    ax.legend()

    # --- Accuracy curves ---
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in metrics['train_acc']], label='Train Accuracy',
            color='#2196F3', linewidth=2)
    ax.plot(epochs, [a * 100 for a in metrics['test_acc']], label='Test Accuracy',
            color='#FF5722', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy')
    ax.set_ylim(-5, 105)
    ax.legend()

    # --- Weight norm ---
    ax = axes[2]
    ax.plot(epochs, metrics['weight_norm'], color='#4CAF50', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sum of Squared Weights')
    ax.set_title('Weight Norm (‖W‖²)')

    fig.suptitle(f"Grokking on Modular Addition (mod {metrics['config']['p']})",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "component1_grokking_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    model, metrics = train_model()
    print("Done! Check figures/ directory for plots.")
