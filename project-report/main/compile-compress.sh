#!/bin/bash
pdflatex -output-directory=../build main.tex
bibtex ../build/main.aux
pdflatex -output-directory=../build main.tex

# compress to screen size (72dpi)
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=../project-report.pdf ../build/main.pdf


