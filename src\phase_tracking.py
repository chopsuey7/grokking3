"""
Component 3: Three Phases of Grokking

Tracks restricted loss, excluded loss, weight norm, and test loss
throughout training to visualize the three phases:
  Phase 1 - Memorization: excluded loss drops
  Phase 2 - Circuit Formation: restricted loss drops, weight norm drops
  Phase 3 - Cleanup: test loss suddenly drops

Usage: python -m src.phase_tracking
"""
import sys, os, torch, torch.nn.functional as F, numpy as np, json
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import OneLayerTransformer
from src.data import make_modular_addition_data
from src.utils import (setup_plotting, compute_accuracy, save_metrics,
                        compute_restricted_loss, compute_excluded_loss,
                        get_key_frequencies, fourier_transform, compute_fourier_power)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = 97
FIGURES_DIR = Path("figures")

def track_phases(p=P, device=None, figures_dir=None, num_epochs=40000, seed=42):
    """Train from scratch while tracking all phase metrics."""
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if figures_dir is None: figures_dir = FIGURES_DIR
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    train_tok, train_lab, test_tok, test_lab = make_modular_addition_data(p, 0.3, seed, device)
    model = OneLayerTransformer(p, 128, 4, 512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

    log_every = 200
    metrics = {k: [] for k in ['epochs','train_loss','test_loss','train_acc','test_acc',
                                 'weight_norm','restricted_loss','excluded_loss']}

    # We'll discover key frequencies dynamically during training
    # Use a running estimate — initially empty, update periodically
    key_freqs = []

    pbar = tqdm(range(num_epochs), desc="Phase tracking")
    for epoch in pbar:
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(train_tok), train_lab)
        loss.backward(); opt.step()

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                tr_logits = model(train_tok); te_logits = model(test_tok)
                tr_loss = F.cross_entropy(tr_logits, train_lab).item()
                te_loss = F.cross_entropy(te_logits, test_lab).item()
                tr_acc = compute_accuracy(tr_logits, train_lab)
                te_acc = compute_accuracy(te_logits, test_lab)
                w_norm = model.sum_squared_weights().item()

                # Update key frequencies every 2000 epochs
                if epoch % 2000 == 0 and epoch > 0:
                    W_E = model.W_E[:p].detach()
                    key_freqs = get_key_frequencies(W_E, p, threshold=0.1)

                # Compute restricted/excluded loss if we have key frequencies
                if key_freqs:
                    r_loss = compute_restricted_loss(model, test_tok, test_lab, key_freqs, p)
                    e_loss = compute_excluded_loss(model, test_tok, test_lab, key_freqs, p)
                else:
                    r_loss = te_loss; e_loss = te_loss

            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(tr_loss)
            metrics['test_loss'].append(te_loss)
            metrics['train_acc'].append(tr_acc)
            metrics['test_acc'].append(te_acc)
            metrics['weight_norm'].append(w_norm)
            metrics['restricted_loss'].append(r_loss)
            metrics['excluded_loss'].append(e_loss)

            pbar.set_postfix(tr=f'{tr_loss:.3f}', te=f'{te_loss:.3f}',
                           acc=f'{te_acc:.1%}', wn=f'{w_norm:.0f}')

    # Plot
    plot_three_phases(metrics, p, figures_dir)
    save_metrics(metrics, str(figures_dir / "phase_tracking_metrics.json"))
    print(f"Saved to {figures_dir}/")
    return metrics

def plot_three_phases(m, p, figures_dir):
    setup_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    e = m['epochs']

    ax = axes[0,0]
    ax.semilogy(e, m['train_loss'], label='Train Loss', color='#2196F3', lw=2)
    ax.semilogy(e, m['test_loss'], label='Test Loss', color='#FF5722', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Standard Losses'); ax.legend()

    ax = axes[0,1]
    ax.semilogy(e, m['restricted_loss'], label='Restricted Loss (key freqs only)', color='#9C27B0', lw=2)
    ax.semilogy(e, m['excluded_loss'], label='Excluded Loss (non-key freqs)', color='#FF9800', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Fourier-Ablated Losses'); ax.legend()

    ax = axes[1,0]
    ax.plot(e, m['weight_norm'], color='#4CAF50', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('‖W‖²'); ax.set_title('Sum of Squared Weights')

    ax = axes[1,1]
    ax.plot(e, [a*100 for a in m['test_acc']], color='#FF5722', lw=2, label='Test Acc')
    ax.plot(e, [a*100 for a in m['train_acc']], color='#2196F3', lw=2, label='Train Acc')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy'); ax.legend()
    ax.set_ylim(-5, 105)

    # Add phase annotations
    for ax in axes.flat:
        ax.axvspan(0, 4000, alpha=0.07, color='red', label='_')
        ax.axvspan(4000, 20000, alpha=0.07, color='blue', label='_')
        ax.axvspan(20000, max(e), alpha=0.07, color='green', label='_')

    # Phase labels on top
    fig.text(0.2, 0.95, '① Memorization', ha='center', fontsize=11, color='red', fontweight='bold')
    fig.text(0.5, 0.95, '② Circuit Formation', ha='center', fontsize=11, color='blue', fontweight='bold')
    fig.text(0.8, 0.95, '③ Cleanup', ha='center', fontsize=11, color='green', fontweight='bold')

    fig.suptitle(f"Three Phases of Grokking (mod {p})", fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "component3_three_phases.png", dpi=150, bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    track_phases()
