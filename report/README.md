# CS 451 Final-Project Report

## Compile (Overleaf, easiest)

1. Create a new Overleaf project, choose "Upload Project", zip this
   `report/` folder and upload.
2. Set the main document to `main.tex`. Overleaf autocompiles.
3. Download `main.pdf` when ready.

## Compile (local, `make`)

```bash
cd report
make           # runs pdflatex → bibtex → pdflatex ×2
```

Produces `main.pdf`. Requires TeX Live (`pdflatex`, `bibtex`).

## Structure

```
report/
├── main.tex         IEEE two-column paper (conference class)
├── references.bib   BibTeX bibliography
├── Makefile         pdflatex + bibtex pipeline
├── README.md        this file
└── figures/         all figures referenced in main.tex (copied from
                     ../figures/ at write time)
```

## Before submitting

- [ ] Fill in Appendix A (AI Usage Log) with entries covering every
      major session. See `../process.MD` for the chronological record.
- [ ] Re-read Introduction and Discussion — prose is drafted but should
      be read in your own voice.
- [ ] Verify all numeric claims against `../docs/data/model_meta.json`.
- [ ] Spot-check figures after recompiling (the figure numbers in the
      prose assume the 12-figure set produced by `src/eda.py` +
      `src/train.py`).
- [ ] Final PDF should be 6–8 pages for CS 451.
