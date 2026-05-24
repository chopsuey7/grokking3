"""
Component 4: Data Fraction Experiments

Varies training data fraction (10%, 30%, 60%, 90%) and plots
"Epochs until Generalization" to show grokking occurs in specific data regimes.

Usage: python -m src.data_fraction
"""
import sys, os, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import OneLayerTransformer
from src.data import make_modular_addition_data
from src.utils import setup_plotting, compute_accuracy, save_metrics
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = 97
FIGURES_DIR = Path("figures")
FRACTIONS = [0.10, 0.30, 0.60, 0.90]
MAX_EPOCHS = 60000
GENERALIZATION_THRESHOLD = 0.95

def train_with_fraction(p, frac, max_epochs, device, seed=42):
    """Train and return epoch of generalization (or None)."""
    torch.manual_seed(seed); np.random.seed(seed)
    train_tok, train_lab, test_tok, test_lab = make_modular_addition_data(p, frac, seed, device)
    model = OneLayerTransformer(p, 128, 4, 512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

    history = {'epochs': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    gen_epoch = None

    for epoch in tqdm(range(max_epochs), desc=f"frac={frac:.0%}", leave=False):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(train_tok), train_lab)
        loss.backward(); opt.step()

        if epoch % 100 == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                tr_acc = compute_accuracy(model(train_tok), train_lab)
                te_acc = compute_accuracy(model(test_tok), test_lab)
                tr_loss = F.cross_entropy(model(train_tok), train_lab).item()
                te_loss = F.cross_entropy(model(test_tok), test_lab).item()
            history['epochs'].append(epoch)
            history['train_loss'].append(tr_loss)
            history['test_loss'].append(te_loss)
            history['train_acc'].append(tr_acc)
            history['test_acc'].append(te_acc)

            if gen_epoch is None and te_acc >= GENERALIZATION_THRESHOLD:
                gen_epoch = epoch

    return gen_epoch, history

def run_data_fraction_experiment(p=P, fractions=None, max_epochs=MAX_EPOCHS,
                                  device=None, figures_dir=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if fractions is None: fractions = FRACTIONS
    if figures_dir is None: figures_dir = FIGURES_DIR
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running data fraction experiment on {device} | p={p}")
    results = {}
    all_histories = {}

    for frac in fractions:
        print(f"\n--- Training with {frac:.0%} data ---")
        gen_epoch, hist = train_with_fraction(p, frac, max_epochs, device)
        results[frac] = gen_epoch
        all_histories[frac] = hist
        if gen_epoch:
            print(f"  Generalized at epoch {gen_epoch}")
        else:
            print(f"  Did NOT generalize within {max_epochs} epochs")

    # Plot
    plot_data_fraction_results(results, all_histories, p, max_epochs, figures_dir)
    save_metrics({'fractions': fractions, 'gen_epochs': {str(k): v for k, v in results.items()},
                  'max_epochs': max_epochs, 'threshold': GENERALIZATION_THRESHOLD},
                 str(figures_dir / "data_fraction_results.json"))
    return results, all_histories

def plot_data_fraction_results(results, histories, p, max_epochs, figures_dir):
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart: Epochs until generalization
    ax = axes[0]
    fracs = sorted(results.keys())
    gen_epochs = [results[f] if results[f] is not None else max_epochs for f in fracs]
    colors = ['#4CAF50' if results[f] is not None else '#F44336' for f in fracs]
    bars = ax.bar([f"{f:.0%}" for f in fracs], gen_epochs, color=colors, alpha=0.8, edgecolor='white', lw=2)
    for bar, f in zip(bars, fracs):
        if results[f] is None:
            bar.set_hatch('///')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                   'No grokking', ha='center', fontsize=9, color='red')
    ax.set_xlabel('Training Data Fraction')
    ax.set_ylabel('Epochs until Generalization')
    ax.set_title(f'Epochs to Generalization vs Data Fraction (mod {p})')

    # Overlay accuracy curves
    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(fracs)))
    for i, f in enumerate(fracs):
        h = histories[f]
        ax.plot(h['epochs'], [a*100 for a in h['test_acc']], label=f"{f:.0%} data",
                color=cmap[i], lw=2)
    ax.axhline(y=GENERALIZATION_THRESHOLD*100, color='red', ls='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Curves by Data Fraction')
    ax.set_ylim(-5, 105); ax.legend()

    fig.suptitle(f"Grokking Depends on Data Scarcity (mod {p})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "component4_data_fraction.png", dpi=150, bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    run_data_fraction_experiment()
