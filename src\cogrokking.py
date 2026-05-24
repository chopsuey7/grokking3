"""
Component 7: Co-Grokking — Multi-task Modular Arithmetic

Trains a single transformer on both addition and multiplication simultaneously.
Demonstrates co-grokking: both tasks generalize at approximately the same time.

Input format: [a, op, b, =] where op is '+' or '×'
Usage: python -m src.cogrokking
"""
import sys, os, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MultiTaskTransformer
from src.data import make_multitask_data
from src.utils import (setup_plotting, compute_accuracy, save_metrics,
                        fourier_transform, compute_fourier_power, get_key_frequencies)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = 97
FIGURES_DIR = Path("figures")

def train_cogrokking(p=P, device=None, figures_dir=None, num_epochs=60000, seed=42):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if figures_dir is None: figures_dir = FIGURES_DIR
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"Co-grokking training on {device} | p={p}")
    data = make_multitask_data(p, train_frac=0.3, seed=seed, device=device)
    tr_tok, tr_lab, tr_ops = data['train_tokens'], data['train_labels'], data['train_ops']
    te_tok, te_lab, te_ops = data['test_tokens'], data['test_labels'], data['test_ops']
    print(f"Train: {len(tr_tok)} ({(tr_ops==0).sum()} add + {(tr_ops==1).sum()} mul)")
    print(f"Test:  {len(te_tok)} ({(te_ops==0).sum()} add + {(te_ops==1).sum()} mul)")

    model = MultiTaskTransformer(p, 128, 4, 512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

    metrics = {k: [] for k in ['epochs','train_loss','test_loss',
        'add_train_acc','add_test_acc','mul_train_acc','mul_test_acc',
        'add_train_loss','add_test_loss','mul_train_loss','mul_test_loss',
        'weight_norm']}

    pbar = tqdm(range(num_epochs), desc="Co-grokking")
    for epoch in pbar:
        model.train(); opt.zero_grad()
        logits = model(tr_tok)
        loss = F.cross_entropy(logits, tr_lab)
        loss.backward(); opt.step()

        if epoch % 200 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                tr_logits = model(tr_tok); te_logits = model(te_tok)

                # Per-task metrics
                add_tr = tr_ops == 0; mul_tr = tr_ops == 1
                add_te = te_ops == 0; mul_te = te_ops == 1

                add_tr_acc = compute_accuracy(tr_logits[add_tr], tr_lab[add_tr])
                add_te_acc = compute_accuracy(te_logits[add_te], te_lab[add_te])
                mul_tr_acc = compute_accuracy(tr_logits[mul_tr], tr_lab[mul_tr])
                mul_te_acc = compute_accuracy(te_logits[mul_te], te_lab[mul_te])

                add_tr_loss = F.cross_entropy(tr_logits[add_tr], tr_lab[add_tr]).item()
                add_te_loss = F.cross_entropy(te_logits[add_te], te_lab[add_te]).item()
                mul_tr_loss = F.cross_entropy(tr_logits[mul_tr], tr_lab[mul_tr]).item()
                mul_te_loss = F.cross_entropy(te_logits[mul_te], te_lab[mul_te]).item()

                total_tr = F.cross_entropy(tr_logits, tr_lab).item()
                total_te = F.cross_entropy(te_logits, te_lab).item()
                wn = model.sum_squared_weights().item()

            metrics['epochs'].append(epoch)
            metrics['train_loss'].append(total_tr); metrics['test_loss'].append(total_te)
            metrics['add_train_acc'].append(add_tr_acc); metrics['add_test_acc'].append(add_te_acc)
            metrics['mul_train_acc'].append(mul_tr_acc); metrics['mul_test_acc'].append(mul_te_acc)
            metrics['add_train_loss'].append(add_tr_loss); metrics['add_test_loss'].append(add_te_loss)
            metrics['mul_train_loss'].append(mul_tr_loss); metrics['mul_test_loss'].append(mul_te_loss)
            metrics['weight_norm'].append(wn)

            pbar.set_postfix(add=f'{add_te_acc:.1%}', mul=f'{mul_te_acc:.1%}')

    # Fourier analysis of shared embeddings
    model.eval()
    W_E = model.W_E[:p].detach()
    key_freqs = get_key_frequencies(W_E, p, threshold=0.1)
    emb_power = compute_fourier_power(W_E, p)

    plot_cogrokking(metrics, emb_power, key_freqs, p, figures_dir)
    save_metrics(metrics, str(figures_dir / "cogrokking_metrics.json"))
    print(f"\nFinal — Add test: {metrics['add_test_acc'][-1]:.1%} | Mul test: {metrics['mul_test_acc'][-1]:.1%}")
    return model, metrics

def plot_cogrokking(m, emb_power, key_freqs, p, figures_dir):
    setup_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    e = m['epochs']

    # Per-task test accuracy
    ax = axes[0,0]
    ax.plot(e, [a*100 for a in m['add_test_acc']], label='Addition (test)', color='#2196F3', lw=2.5)
    ax.plot(e, [a*100 for a in m['mul_test_acc']], label='Multiplication (test)', color='#FF5722', lw=2.5)
    ax.plot(e, [a*100 for a in m['add_train_acc']], label='Addition (train)', color='#2196F3', lw=1, ls='--', alpha=0.5)
    ax.plot(e, [a*100 for a in m['mul_train_acc']], label='Multiplication (train)', color='#FF5722', lw=1, ls='--', alpha=0.5)
    ax.axhline(y=95, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Co-Grokking: Per-Task Accuracy'); ax.legend(); ax.set_ylim(-5, 105)

    # Per-task test loss
    ax = axes[0,1]
    ax.semilogy(e, m['add_test_loss'], label='Addition (test)', color='#2196F3', lw=2)
    ax.semilogy(e, m['mul_test_loss'], label='Multiplication (test)', color='#FF5722', lw=2)
    ax.semilogy(e, m['add_train_loss'], label='Addition (train)', color='#2196F3', lw=1, ls='--', alpha=0.5)
    ax.semilogy(e, m['mul_train_loss'], label='Multiplication (train)', color='#FF5722', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Co-Grokking: Per-Task Loss'); ax.legend()

    # Weight norm
    ax = axes[1,0]
    ax.plot(e, m['weight_norm'], color='#4CAF50', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('‖W‖²'); ax.set_title('Weight Norm')

    # Shared embedding Fourier power
    ax = axes[1,1]
    n_freq = (p+1)//2; freq_idx = np.arange(n_freq)
    colors = ['#FF5722' if f in key_freqs else '#90CAF9' for f in freq_idx]
    ax.bar(freq_idx, emb_power.cpu().numpy(), color=colors, alpha=0.8)
    ax.set_xlabel('Frequency k'); ax.set_ylabel('Power')
    ax.set_title('Shared Embedding Fourier Power')

    fig.suptitle(f"Co-Grokking: Simultaneous Generalization (mod {p})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "component7_cogrokking.png", dpi=150, bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    train_cogrokking()
