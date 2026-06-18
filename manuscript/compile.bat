@echo off
cd /d "%~dp0"
..\.tools\tinytex\dist\TinyTeX\bin\windows\pdflatex.exe -interaction=nonstopmode main.tex
