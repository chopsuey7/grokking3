"""
Component 2: Fourier Analysis of Embeddings and MLP Activations

Demonstrates that the network learns sparse key Fourier frequencies.
Usage: python -m src.fourier_analysis
"""
import sys, os, torch, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import OneLayerTransformer
from src.data import make_modular_addition_data
from src.utils import (setup_plotting, fourier_basis, fourier_transform,
                        compute_fourier_power, get_key_frequencies, save_metrics)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = 97
FIGURES_DIR = Path("figures")
CHECKPOINTS_DIR = Path("checkpoints")

def find_latest_checkpoint(d):
    ckpts = sorted(Path(d).glob("model_epoch_*.pt"))
    if not ckpts: raise FileNotFoundError(f"No checkpoints in {d}")
    return ckpts[-1]

def analyze_fourier(model=None, p=P, device=None, figures_dir=None, checkpoints_dir=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if figures_dir is None: figures_dir = FIGURES_DIR
    if checkpoints_dir is None: checkpoints_dir = CHECKPOINTS_DIR
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        ckpt_path = find_latest_checkpoint(checkpoints_dir)
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model = OneLayerTransformer(cfg['p'], cfg['d_model'], cfg['n_heads'], cfg['d_mlp']).to(device)
        model.load_state_dict(ckpt['model_state_dict']); p = cfg['p']
    model.eval()

    # 1. Embedding Fourier Analysis
    W_E = model.W_E[:p].detach()
    W_E_fourier = fourier_transform(W_E, p)
    embedding_power = compute_fourier_power(W_E, p)
    key_freqs = get_key_frequencies(W_E, p, threshold=0.1)
    print(f"Key frequencies: {key_freqs} ({len(key_freqs)} of {(p+1)//2})")

    # 2. MLP Activation Fourier Analysis
    all_a = torch.arange(p, device=device).repeat_interleave(p)
    all_b = torch.arange(p, device=device).repeat(p)
    all_tokens = torch.stack([all_a, all_b, torch.full_like(all_a, p)], dim=1)
    with torch.no_grad():
        _, cache = model(all_tokens, return_cache=True)
    mlp_post = cache['mlp_post'][:, -1, :].reshape(p, p, -1)
    mlp_vs_a = mlp_post.mean(dim=1)
    mlp_vs_b = mlp_post.mean(dim=0)
    mlp_power_a = compute_fourier_power(mlp_vs_a, p)
    mlp_power_b = compute_fourier_power(mlp_vs_b, p)
    mlp_fourier_a = fourier_transform(mlp_vs_a, p)

    # Plotting
    setup_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    n_freq = (p + 1) // 2; freq_idx = np.arange(n_freq)
    colors = ['#FF5722' if f in key_freqs else '#90CAF9' for f in freq_idx]

    axes[0,0].bar(freq_idx, embedding_power.cpu().numpy(), color=colors, alpha=0.8)
    axes[0,0].set_xlabel('Frequency k'); axes[0,0].set_ylabel('Power')
    axes[0,0].set_title('W_E Fourier Power Spectrum')

    im = axes[0,1].imshow(np.abs(W_E_fourier.cpu().numpy()), aspect='auto', cmap='viridis')
    axes[0,1].set_xlabel('Embedding Dim'); axes[0,1].set_ylabel('Fourier Component')
    axes[0,1].set_title('|DFT(W_E)| Heatmap'); plt.colorbar(im, ax=axes[0,1])

    sp, si = embedding_power.sort(descending=True)
    top_k = min(20, len(freq_idx)); tf = si[:top_k].cpu().numpy(); tp = sp[:top_k].cpu().numpy()
    axes[0,2].barh(range(top_k), tp, color=['#FF5722' if f in key_freqs else '#90CAF9' for f in tf])
    axes[0,2].set_yticks(range(top_k)); axes[0,2].set_yticklabels([f'k={f}' for f in tf])
    axes[0,2].set_title(f'Top {top_k} Frequencies'); axes[0,2].invert_yaxis()

    axes[1,0].bar(freq_idx, mlp_power_a.cpu().numpy(), color=colors, alpha=0.8)
    axes[1,0].set_xlabel('Frequency k'); axes[1,0].set_title('MLP Activations (vs a)')

    axes[1,1].bar(freq_idx, mlp_power_b.cpu().numpy(), color=colors, alpha=0.8)
    axes[1,1].set_xlabel('Frequency k'); axes[1,1].set_title('MLP Activations (vs b)')

    n_show = min(64, mlp_fourier_a.shape[1])
    im2 = axes[1,2].imshow(np.abs(mlp_fourier_a[:,:n_show].cpu().numpy()), aspect='auto', cmap='magma')
    axes[1,2].set_xlabel(f'Neuron (first {n_show})'); axes[1,2].set_ylabel('Fourier Component')
    axes[1,2].set_title('|DFT(MLP)| per Neuron'); plt.colorbar(im2, ax=axes[1,2])

    fig.suptitle(f"Fourier Analysis — Sparse Key Frequencies (mod {p})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "component2_fourier_analysis.png", dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved to {figures_dir}/component2_fourier_analysis.png")

    save_metrics({'key_frequencies': key_freqs, 'embedding_power': embedding_power.cpu().tolist(),
        'mlp_power_vs_a': mlp_power_a.cpu().tolist(), 'mlp_power_vs_b': mlp_power_b.cpu().tolist()},
        str(figures_dir / "fourier_analysis_results.json"))
    return key_freqs, embedding_power

if __name__ == "__main__":
    key_freqs, power = analyze_fourier()
    print(f"\nKey frequencies: {key_freqs}")
