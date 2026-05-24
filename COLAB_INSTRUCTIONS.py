"""
=============================================================
  GROKKING PROJECT — GOOGLE COLAB INSTRUCTIONS
=============================================================

STEP-BY-STEP:

1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook
3. Change runtime to GPU: Runtime → Change runtime type → T4 GPU
4. Copy-paste the cells below IN ORDER, and run each one

=============================================================
"""

# %%
# ═══════════════════════════════════════════════════════════
# CELL 1: Upload & Setup (run this first, then upload the zip when prompted)
# ═══════════════════════════════════════════════════════════

import os
from google.colab import files

# Upload the zip file
print("📁 Upload grokking_project.zip when prompted...")
uploaded = files.upload()

# Unzip
!mkdir -p /content/grokking
!unzip -o /content/grokking_project.zip -d /content/grokking
os.chdir('/content/grokking')

# Install dependencies
!pip install torch numpy matplotlib scipy einops tqdm -q

# Verify
!python -c "from src.model import OneLayerTransformer; import torch; print('✅ Setup complete! GPU:', torch.cuda.is_available())"


# %%
# ═══════════════════════════════════════════════════════════
# CELL 2: Run Component 1 — Reproduce Grokking (~10 min)
# ═══════════════════════════════════════════════════════════

!python run_all.py --components 1


# %%
# ═══════════════════════════════════════════════════════════
# CELL 3: Run Component 2 — Fourier Analysis (~30 sec)
# ═══════════════════════════════════════════════════════════

!python run_all.py --components 2


# %%
# ═══════════════════════════════════════════════════════════
# CELL 4: Run Component 3 — Three Phases (~10 min)
# ═══════════════════════════════════════════════════════════

!python run_all.py --components 3


# %%
# ═══════════════════════════════════════════════════════════
# CELL 5: Run Component 4 — Data Fraction (~40 min)
# ═══════════════════════════════════════════════════════════

!python run_all.py --components 4


# %%
# ═══════════════════════════════════════════════════════════
# CELL 6: Run Component 6 — Modular Multiplication (~12 min)
# ═══════════════════════════════════════════════════════════

!python run_all.py --components 6


# %%
# ═══════════════════════════════════════════════════════════
# CELL 7: Run Component 7 — Co-Grokking (~15 min)
# ═══════════════════════════════════════════════════════════

!python run_all.py --components 7


# %%
# ═══════════════════════════════════════════════════════════
# CELL 8: Build Report & Embed Figures
# ═══════════════════════════════════════════════════════════

!python run_all.py --components 8

# Show what figures were generated
import os
print("\n📊 Generated figures:")
for f in sorted(os.listdir('figures')):
    if f.endswith('.png'):
        print(f"  ✓ {f}")


# %%
# ═══════════════════════════════════════════════════════════
# CELL 9: Preview Figures (optional — view in notebook)
# ═══════════════════════════════════════════════════════════

from IPython.display import display, Image
import os

for f in sorted(os.listdir('figures')):
    if f.endswith('.png'):
        print(f"\n{'='*60}")
        print(f"  {f}")
        print(f"{'='*60}")
        display(Image(filename=f'figures/{f}', width=800))


# %%
# ═══════════════════════════════════════════════════════════
# CELL 10: Download Everything for GitHub
# ═══════════════════════════════════════════════════════════

# Zip the docs folder (for GitHub Pages) and figures
!cd /content/grokking && zip -r /content/github_pages.zip docs/
!cd /content/grokking && zip -r /content/full_project.zip .

from google.colab import files

print("📥 Downloading github_pages.zip (just the report site)...")
files.download('/content/github_pages.zip')

print("📥 Downloading full_project.zip (entire project with code)...")
files.download('/content/full_project.zip')

print("""
✅ DONE! Now for GitHub Pages:

1. Go to github.com → New Repository → name it "grokking"
2. Unzip full_project.zip into the repo folder
3. git add . && git commit -m "Grokking project" && git push
4. Go to Settings → Pages → Source: main branch, /docs folder → Save
5. Your report will be live at: https://YOUR_USERNAME.github.io/grokking/
""")
