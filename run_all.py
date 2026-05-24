"""
Master Runner — Run all grokking project components.

This script can be run in Google Colab or locally.
Each component can also be run independently.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --components 1 2   # Run specific components
"""
import sys, os, argparse, time
from pathlib import Path

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_component(num, device='cuda'):
    start = time.time()
    print(f"\n{'='*60}")
    print(f"  COMPONENT {num}")
    print(f"{'='*60}\n")

    if num == 1:
        from src.train import train_model
        model, metrics = train_model(device=device)
        return model

    elif num == 2:
        from src.fourier_analysis import analyze_fourier
        key_freqs, power = analyze_fourier(device=device)

    elif num == 3:
        from src.phase_tracking import track_phases
        track_phases(device=device)

    elif num == 4:
        from src.data_fraction import run_data_fraction_experiment
        run_data_fraction_experiment(device=device)

    elif num == 5:
        print("Component 5 (Alignment Essay) is pre-written.")
        print("See report/alignment_essay.md")

    elif num == 6:
        from src.multiplication import train_multiplication
        train_multiplication(device=device)

    elif num == 7:
        from src.cogrokking import train_cogrokking
        train_cogrokking(device=device)

    elif num == 8:
        from report.build_report import build_report
        build_report()

    elapsed = time.time() - start
    print(f"\nComponent {num} completed in {elapsed:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Grokking Project Runner")
    parser.add_argument('--components', nargs='+', type=int, default=None,
                       help='Which components to run (default: all)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu, default: auto-detect)')
    args = parser.parse_args()

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    components = args.components or [1, 2, 3, 4, 5, 6, 7, 8]

    total_start = time.time()
    for c in components:
        run_component(c, device=device)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")
    print(f"Figures:     figures/")
    print(f"Checkpoints: checkpoints/")
    print(f"Report:      docs/index.html")

if __name__ == "__main__":
    main()
