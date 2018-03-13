texfiles := $(wildcard *.tex)

all: pdf

pdf: main.pdf

main.pdf: $(texfiles) offshore.bib
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -rf auto *.bbl *.blg *.ps *.dvi *.aux *.toc *.spl \
		*.idx *.ind *.ilg *.log *.out main.pdf *~ \
		*.lof *.lot *.fff *.ttt *.upa *.upb

