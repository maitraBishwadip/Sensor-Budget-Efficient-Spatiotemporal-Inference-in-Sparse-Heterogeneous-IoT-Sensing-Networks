# IEEE TGRS manuscript — build

- `main.tex` — the manuscript (IEEEtran `journal`, two-column). Bibliography is **embedded** (`thebibliography`), so **no BibTeX run is needed**.
- `IEEEtran.cls` — the IEEE class file (copied here for a self-contained build).
- Figures are pulled from `../paper_figs/*.pdf` via `\graphicspath`.

## Compile
No LaTeX toolchain is installed locally. Two options:

**Overleaf (recommended):** upload `paper/` and `paper_figs/` (keep the relative path `../paper_figs/`), or flatten figures into `paper/`. Compile with pdfLaTeX, run twice for cross-references.

**Local TeX (after installing MiKTeX/TeX Live):**
```
cd paper
pdflatex main
pdflatex main
```

## Regenerate figures
```
python paper_figs/export_predictions.py   # real predictions for Fig. 6 (verifies R2=0.8158)
python paper_figs/make_figures.py          # all 6 figures (PDF + PNG)
```
