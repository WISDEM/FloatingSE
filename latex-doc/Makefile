texfiles := $(wildcard *.tex)

all: pdf

pdf: main.pdf

main.pdf: $(texfiles) offshore.bib
	latexmk -pdf main.tex

clean:
	rm -rf auto *.bbl *.blg *.ps *.dvi *.aux *.toc *.spl \
		*.idx *.ind *.ilg *.log *.out main.pdf *~ \
		*.lof *.lot *.fff *.ttt *.upa *.upb

