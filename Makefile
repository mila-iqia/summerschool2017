theano.pdf: slides_source/theano.tex
	cd slides_source; pdflatex --shell-escape theano.tex
	mv slides_source/theano.pdf .
