.PHONY: all figures clean
all:
	xelatex -interaction=nonstopmode -halt-on-error report.tex
	bibtex report
	xelatex -interaction=nonstopmode -halt-on-error report.tex
	xelatex -interaction=nonstopmode -halt-on-error report.tex

figures:
	python build_figures.py

clean:
	$(RM) report.aux report.bbl report.blg report.log report.out report.toc report.synctex.gz
