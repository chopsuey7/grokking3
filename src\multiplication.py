"""
Component 6: Modular Multiplication / Exponentiation Algorithm

Trains a one-layer transformer on (a * b) mod p, reproduces grokking,
and analyzes the learned algorithm via Fourier analysis.

The multiplication algorithm differs from addition:
- Addition uses cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb)
- Multiplication may use discrete log: a*b = g^(log_g(a) + log_g(b))

Usage: python -m src.multiplication
"""
import sys, os, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import OneLayerTransformer
from src.data import make_modular_multiplication_data
from src.utils import (setup_plotting, compute_accuracy, save_metrics,
                        fourier_transform, compute_fourier_power, get_key_frequencies)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = 97
FIGURES_DIR = Path("figures")
CHECKPOINTS_DIR = Path("checkpoints")

def find_primitive_root(p):
    """Find the smallest primitive root modulo p."""
    for g in range(2, p):
        seen = set()
        val = 1
        for _ in range(p - 1):
            val = (val * g) % p
            seen.add(val)
        if len(seen) == p - 1:
            return g
    return None

def train_multiplication(p=P, device=None, figures_dir=None, num_epochs=50000, seed=42):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if figures_dir is None: figures_dir = FIGURES_DIR
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"Training modular multiplication on {device} | p={p}")
    train_tok, train_lab, test_tok, test_lab = make_modular_multiplication_data(p, 0.3, seed, device)
    print(f"Train: {len(train_tok)} | Test: {len(test_tok)}")

    model = OneLayerTransformer(p, 128, 4, 512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

    metrics = {k: [] for k in ['epochs','train_loss','test_loss','train_acc','test_acc','weight_norm']}

    pbar = tqdm(range(num_epochs), desc="Multiplication training")
    for epoch in pbar:
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(train_tok), train_lab)
        loss.backward(); opt.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                tr_l = F.cross_entropy(model(train_tok), train_lab).item()
                te_l = F.cross_entropy(model(test_tok), test_lab).item()
                tr_a = compute_accuracy(model(train_tok), train_lab)
                te_a = compute_accuracy(model(test_tok), test_lab)
                wn = model.sum_squared_weights().item()
            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(tr_l); metrics['test_loss'].append(te_l)
            metrics['train_acc'].append(tr_a); metrics['test_acc'].append(te_a)
            metrics['weight_norm'].append(wn)
            pbar.set_postfix(tr=f'{tr_l:.3f}', te=f'{te_l:.3f}', acc=f'{te_a:.1%}')

    # Fourier analysis of multiplication model
    print("\nFourier analysis of multiplication embeddings...")
    model.eval()
    W_E = model.W_E[:p].detach()
    embedding_power = compute_fourier_power(W_E, p)
    key_freqs = get_key_frequencies(W_E, p, threshold=0.1)
    print(f"Key frequencies (multiplication): {key_freqs}")

    # Discrete log analysis
    g = find_primitive_root(p)
    print(f"Primitive root g={g} for p={p}")

    # Check if embeddings are organized by discrete log
    dlog = np.zeros(p, dtype=int)
    val = 1
    for i in range(p - 1):
        dlog[val] = i
        val = (val * g) % p
    # dlog[0] is undefined for multiplication, set to -1
    dlog_order = np.argsort(dlog[1:]) + 1  # reorder tokens 1..p-1 by their discrete log

    # Plotting
    plot_multiplication_results(metrics, embedding_power, key_freqs, W_E, dlog, dlog_order, p, g, figures_dir)
    save_metrics({**metrics, 'key_frequencies_mul': key_freqs,
                  'primitive_root': g}, str(figures_dir / "multiplication_metrics.json"))
    return model, metrics

def plot_multiplication_results(m, power, key_freqs, W_E, dlog, dlog_order, p, g, figures_dir):
    setup_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    e = m['epochs']

    # Training curves
    axes[0,0].semilogy(e, m['train_loss'], label='Train', color='#2196F3', lw=2)
    axes[0,0].semilogy(e, m['test_loss'], label='Test', color='#FF5722', lw=2)
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Multiplication: Loss Curves'); axes[0,0].legend()

    axes[0,1].plot(e, [a*100 for a in m['train_acc']], label='Train', color='#2196F3', lw=2)
    axes[0,1].plot(e, [a*100 for a in m['test_acc']], label='Test', color='#FF5722', lw=2)
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].set_title('Multiplication: Accuracy'); axes[0,1].legend(); axes[0,1].set_ylim(-5,105)

    axes[0,2].plot(e, m['weight_norm'], color='#4CAF50', lw=2)
    axes[0,2].set_xlabel('Epoch'); axes[0,2].set_ylabel('‖W‖²')
    axes[0,2].set_title('Weight Norm')

    # Fourier power
    n_freq = (p+1)//2; freq_idx = np.arange(n_freq)
    colors = ['#FF5722' if f in key_freqs else '#90CAF9' for f in freq_idx]
    axes[1,0].bar(freq_idx, power.cpu().numpy(), color=colors, alpha=0.8)
    axes[1,0].set_xlabel('Frequency k'); axes[1,0].set_title('Embedding Fourier Power (Multiplication)')

    # PCA of embeddings colored by discrete log
    from torch.linalg import svd
    W_num = W_E[1:].cpu()  # exclude token 0 (multiplication by 0 = 0)
    U, S, Vt = svd(W_num - W_num.mean(dim=0))
    pcs = (U * S)[:, :2].numpy()
    sc = axes[1,1].scatter(pcs[:,0], pcs[:,1], c=dlog[1:], cmap='hsv', s=20, alpha=0.7)
    axes[1,1].set_xlabel('PC1'); axes[1,1].set_ylabel('PC2')
    axes[1,1].set_title('Embeddings colored by discrete log')
    plt.colorbar(sc, ax=axes[1,1], label='log_g(token)')

    # Embedding norms vs discrete log
    norms = W_num.norm(dim=1).numpy()
    axes[1,2].scatter(dlog[1:], norms, s=20, alpha=0.7, color='#9C27B0')
    axes[1,2].set_xlabel('Discrete log'); axes[1,2].set_ylabel('Embedding norm')
    axes[1,2].set_title(f'Embedding Norm vs log_{{g={g}}}(token)')

    fig.suptitle(f"Modular Multiplication Algorithm Analysis (mod {p})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "component6_multiplication.png", dpi=150, bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    train_multiplication()
