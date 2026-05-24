"""
Component 8: Embed Figures into GitHub Pages Report

After running all training components, run this script to replace
the figure placeholders in docs/index.html with actual base64-encoded images.

Usage: python -m report.build_report   (from grokking/ directory)
"""
import sys, os, base64, re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR = Path("figures")
DOCS_DIR = Path("docs")

FIGURE_MAP = {
    'fig-c1': 'component1_grokking_curves.png',
    'fig-c2': 'component2_fourier_analysis.png',
    'fig-c3': 'component3_three_phases.png',
    'fig-c4': 'component4_data_fraction.png',
    'fig-c6': 'component6_multiplication.png',
    'fig-c7': 'component7_cogrokking.png',
}

def img_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def build_report(figures_dir=None, docs_dir=None):
    if figures_dir is None: figures_dir = FIGURES_DIR
    if docs_dir is None: docs_dir = DOCS_DIR
    figures_dir = Path(figures_dir); docs_dir = Path(docs_dir)

    html_path = docs_dir / "index.html"
    if not html_path.exists():
        print(f"Error: {html_path} not found. The HTML template should already exist.")
        return

    html = html_path.read_text(encoding='utf-8')
    replaced = 0

    for fig_id, filename in FIGURE_MAP.items():
        fig_path = figures_dir / filename
        if fig_path.exists():
            b64 = img_to_base64(fig_path)
            # Replace the placeholder div with an actual image
            pattern = (
                rf'<div class="figure-placeholder" id="{fig_id}">'
                r'.*?</div>'
            )
            img_tag = f'<img src="data:image/png;base64,{b64}" alt="{fig_id}" style="width:100%;display:block">'
            html, n = re.subn(pattern, img_tag, html, flags=re.DOTALL)
            if n > 0:
                replaced += 1
                print(f"  [OK] Embedded {filename} -> #{fig_id}")
            else:
                print(f"  [WARN] Could not find placeholder #{fig_id}")
        else:
            print(f"  [SKIP] {filename} (not yet generated)")

    html_path.write_text(html, encoding='utf-8')
    print(f"\nEmbedded {replaced}/{len(FIGURE_MAP)} figures into {html_path}")
    return html_path

if __name__ == "__main__":
    build_report()
