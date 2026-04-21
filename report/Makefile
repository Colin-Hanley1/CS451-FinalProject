# Compile the CS 451 final-project report (IEEEtran, two-column).
# Requires pdflatex + bibtex (TeX Live or MiKTeX). Simplest path is
# to paste main.tex + references.bib + figures/ into Overleaf.

TEX   = pdflatex
BIB   = bibtex
MAIN  = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(TEX) -interaction=nonstopmode $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) -interaction=nonstopmode $(MAIN).tex
	$(TEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).synctex.gz
