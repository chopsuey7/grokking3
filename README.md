# The "Aha!" Moment: Understanding Grokking in AI

Investigating grokking phenomena using a one-layer transformer on modular arithmetic, through the lens of mechanistic interpretability.

## 🔗 [View the Report](https://YOUR_USERNAME.github.io/grokking/)

## Components

| # | Component | Marks |
|---|-----------|-------|
| 1 | Reproduce grokking on modular addition (mod 97) | 10 |
| 2 | Fourier analysis of embeddings & MLP activations | +5 |
| 3 | Three phases: Memorization → Circuit Formation → Cleanup | +5 |
| 4 | Data fraction experiments (10%, 30%, 60%, 90%) | +5 |
| 5 | AI Alignment & Mechanistic Interpretability essay | +5 |
| 6 | Modular multiplication algorithm analysis | +10 |
| 7 | Co-grokking: multi-task simultaneous generalization | +15 |
| 8 | GitHub Pages report | +5 |

## Running

```bash
pip install torch numpy matplotlib scipy einops tqdm
python run_all.py              # Run all components
python run_all.py --components 1 2  # Run specific ones
```

## References

- Nanda et al., *Progress measures for grokking via mechanistic interpretability* (ICLR 2023)
- Power et al., *Grokking: Generalization beyond overfitting on small algorithmic datasets* (ICLR 2022)
