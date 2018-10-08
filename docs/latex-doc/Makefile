texfiles := $(wildcard *.tex)

all: pdf

pdf: main.pdf

main.pdf: $(texfiles) offshore.bib
	latexmk -pdf main.tex

clean:
	rm -rf auto *.bbl *.blg *.ps *.dvi *.aux *.toc *.spl *.xml\
		*.idx *.ind *.ilg *.log *.out main.pdf *~ *blx.bib\
		*.lof *.lot *.fff *.ttt *.upa *.upb *.fls *.fdb_latexmk

