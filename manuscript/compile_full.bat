@echo off
cd /d "%~dp0"
set BIN=..\.tools\tinytex\dist\TinyTeX\bin\windows
echo === pdflatex pass 1 ===
%BIN%\pdflatex.exe -interaction=nonstopmode main.tex > nul 2>&1
echo === bibtex ===
%BIN%\bibtex.exe main
echo === pdflatex pass 2 ===
%BIN%\pdflatex.exe -interaction=nonstopmode main.tex > nul 2>&1
echo === pdflatex pass 3 ===
%BIN%\pdflatex.exe -interaction=nonstopmode main.tex 2>&1 | findstr /c:"Output" /c:"undefined citations" /c:"Fatal" /c:"Warning.*There were"
